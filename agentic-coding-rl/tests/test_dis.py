"""DIS numerical/gradient oracles and the Miles custom-loss CPU contract."""
import argparse
import copy
import math
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / ".deps/miles")]
from algorithms.dis_core import dis_policy_terms
from algorithms import grpo_dis
from launcher import read_config, runtime_env
from native_cpu import native_functions


class DISMathTests(unittest.TestCase):
    def test_two_sides_are_masked_for_both_advantage_signs(self):
        ratio = torch.tensor([0.1, 0.3, 1., 3., 5.], dtype=torch.float64)
        for sign in (-1, 1):
            logp = ratio.log().requires_grad_()
            result = dis_policy_terms(logp, torch.zeros_like(logp), torch.full_like(logp, sign), torch.ones_like(logp))
            result.policy_loss.sum().backward()
            oracle = torch.tensor([0., .3, 1., 3., 0.], dtype=torch.float64)
            torch.testing.assert_close(result.ratio_weight, oracle)
            torch.testing.assert_close(logp.grad, -sign * oracle)

    def test_strict_endpoints_for_float32_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            logp = torch.tensor([math.log(.2), math.log(4.)], dtype=dtype, requires_grad=True)
            result = dis_policy_terms(logp, torch.zeros_like(logp), torch.ones_like(logp), torch.ones_like(logp))
            self.assertFalse(result.kept.any())
            result.policy_loss.sum().backward()
            torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

    def test_ratio_gradient_matches_score_function_not_weight_times_logp(self):
        logp = torch.tensor([-2.1, -2.3, -1.7], dtype=torch.float64, requires_grad=True)
        logq = torch.tensor([-2.2, -2., -2.], dtype=torch.float64, requires_grad=True)
        advantage = torch.tensor([2., -3., .5], dtype=torch.float64, requires_grad=True)
        result = dis_policy_terms(logp, logq, advantage, torch.ones_like(logp))
        result.policy_loss.sum().backward()
        expected = -(logp.detach() - logq.detach()).exp() * advantage.detach()
        torch.testing.assert_close(logp.grad, expected)
        self.assertIsNone(logq.grad)
        self.assertIsNone(advantage.grad)

    def test_finite_difference_gradcheck_away_from_mask_boundaries(self):
        logp = torch.tensor([-3., -2., -1.], dtype=torch.float64, requires_grad=True)
        logq = torch.full_like(logp, -2.)
        advantage = torch.tensor([1., -2., .3], dtype=torch.float64)
        fn = lambda x: dis_policy_terms(x, logq, advantage, torch.ones_like(x)).policy_loss.sum()
        self.assertTrue(torch.autograd.gradcheck(fn, (logp,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_extreme_ratios_never_exponentiate_rejected_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            logp = torch.tensor([-10000., 0., 10000.], dtype=dtype, requires_grad=True)
            result = dis_policy_terms(logp, torch.zeros_like(logp), torch.ones_like(logp), torch.ones_like(logp))
            result.policy_loss.sum().backward()
            self.assertTrue(torch.isfinite(result.policy_loss).all())
            torch.testing.assert_close(logp.grad, torch.tensor([0., -1., 0.], dtype=dtype))

    def test_observations_with_nan_are_sanitized_before_math(self):
        logp = torch.tensor([0., float("nan"), float("inf")], requires_grad=True)
        result = dis_policy_terms(logp, torch.tensor([0., float("inf"), float("nan")]),
                                  torch.tensor([2., float("nan"), float("inf")]), torch.tensor([1., 0., 0.]))
        result.policy_loss.sum().backward()
        torch.testing.assert_close(result.policy_loss.detach(), torch.tensor([-2., 0., 0.]))
        torch.testing.assert_close(logp.grad, torch.tensor([-2., 0., 0.]))

    def test_nonfinite_active_values_fail_instead_of_being_silently_dropped(self):
        for position in range(3):
            for bad in (float("nan"), float("inf"), -float("inf")):
                values = [torch.zeros(1), torch.zeros(1), torch.ones(1), torch.ones(1)]
                values[position][0] = bad
                with self.assertRaises(FloatingPointError):
                    dis_policy_terms(*values)

    def test_empty_all_masked_and_all_rejected_are_differentiable(self):
        for logp, mask in ((torch.empty(0), torch.empty(0)), (torch.ones(3), torch.zeros(3)),
                           (torch.full((3,), 1000.), torch.ones(3))):
            logp.requires_grad_()
            result = dis_policy_terms(logp, torch.zeros_like(logp), torch.ones_like(logp), mask)
            result.policy_loss.sum().backward()
            torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

    def test_invalid_shapes_masks_and_bounds_are_rejected(self):
        x = torch.zeros(2)
        for low, high in ((0., 4.), (.2, 1.), (1., 4.), (.2, float("inf")), (float("nan"), 4.)):
            with self.assertRaises(ValueError):
                dis_policy_terms(x, x, x, torch.ones_like(x), lower=low, upper=high)
        with self.assertRaises(ValueError):
            dis_policy_terms(x, x[:1], x, x)
        with self.assertRaises(ValueError):
            dis_policy_terms(x, x, x, torch.tensor([.5, 1.]))

    def test_real_softmax_optimizer_step_matches_independent_is_oracle(self):
        generator = torch.Generator().manual_seed(117)
        logits = torch.randn(6, 5, dtype=torch.float64, generator=generator, requires_grad=True)
        oracle_logits = logits.detach().clone().requires_grad_()
        tokens = torch.tensor([0, 1, 2, 3, 4, 0])
        logp = logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
        logq = logp.detach() - torch.tensor([-3., -.4, 0., .5, 2., -.5], dtype=torch.float64)
        a = torch.tensor([1., -2., .5, -1., 2., 3.], dtype=torch.float64)
        mask = torch.tensor([1., 1., 0., 1., 1., 1.], dtype=torch.float64)
        loss = dis_policy_terms(logp, logq, a, mask).policy_loss.sum() / mask.sum()
        oracle_p = oracle_logits.softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
        ratio = oracle_p / logq.exp()
        oracle = -(ratio * a * ((ratio > .2) & (ratio < 4.)) * mask).sum() / mask.sum()
        loss.backward(); oracle.backward()
        torch.testing.assert_close(logits.grad, oracle_logits.grad)
        torch.testing.assert_close(logits.detach() - .01 * logits.grad,
                                   oracle_logits.detach() - .01 * oracle_logits.grad)

    def test_randomized_closed_form_softmax_gradient_oracle(self):
        # Independent closed-form categorical-policy gradient: no autograd and
        # no call to the production ratio/mask code on the oracle side.
        generator = torch.Generator().manual_seed(20261029)
        for dtype in (torch.float32, torch.float64):
            for case in range(50):
                count, vocab = 3 + case % 31, 2 + case % 13
                logits = (2 * torch.randn(count, vocab, generator=generator, dtype=dtype)).requires_grad_()
                old = 2 * torch.randn(count, vocab, generator=generator, dtype=dtype)
                tokens = torch.randint(vocab, (count,), generator=generator)
                advantage = torch.randn(count, generator=generator, dtype=dtype)
                mask = torch.randint(2, (count,), generator=generator).to(dtype)
                mask[0] = 1
                logp = logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
                logq = old.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
                result = dis_policy_terms(logp, logq, advantage, mask)
                (result.policy_loss.sum() / mask.sum()).backward()
                probability = logits.detach().softmax(-1)
                chosen = probability.gather(-1, tokens[:, None]).squeeze(-1)
                behavior = old.softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
                ratio = chosen / behavior
                valid = mask.bool() & (ratio > .2) & (ratio < 4.)
                one_hot = torch.nn.functional.one_hot(tokens, vocab).to(dtype)
                expected = -(valid * ratio * advantage / mask.sum())[:, None] * (one_hot - probability)
                with self.subTest(dtype=dtype, case=case):
                    torch.testing.assert_close(logits.grad, expected, atol=1e-6 if dtype == torch.float32 else 1e-12,
                                               rtol=1e-5 if dtype == torch.float32 else 1e-10)


class DISAdapterTests(unittest.TestCase):
    def setUp(self):
        self.parallel = NS(cp=NS(size=1, rank=0), intra_dp=NS(size=1), intra_dp_cp=NS(size=1), is_ulysses_cp=False)
        self.cp = native_functions("miles/backends/training_utils/data/context_parallel.py",
            ("get_sum_of_sample_mean", "get_local_response_loss_masks", "slice_log_prob_with_cp"),
            get_parallel_state=lambda: self.parallel)
        self.math = native_functions("miles/backends/training_utils/loss/hub/math_utils.py", ("compute_approx_kl",))
        self.args = NS(advantage_estimator="grpo", use_rollout_logprobs=True, use_tis=False,
            custom_tis_function_path=None, use_opsm=False, custom_pg_loss_reducer_function_path=None,
            use_sampling_support_replay=False, use_unbiased_kl=False, use_opd=False,
            rollout_top_p=1., rollout_top_k=-1, entropy_coef=0., observe_training_entropy=False,
            qkv_format="thd", use_kl_loss=False, kl_loss_coef=0., kl_loss_type="low_var_kl",
            dis_ratio_low=.2, dis_ratio_high=4., backend=NS(calculate_per_token_loss=False, global_batch_size=2))
        self.batch = {"unconcat_tokens": [torch.tensor([0, 1, 2, 1]), torch.tensor([1, 2])],
            "total_lengths": [4, 2], "response_lengths": [3, 1],
            "loss_masks": [torch.tensor([1., 0., 1.]), torch.tensor([1.])],
            "advantages": [torch.tensor([1., 9., -1.]), torch.tensor([2.])],
            "rollout_log_probs": [torch.tensor([-2., float("nan"), -2.]), torch.tensor([-4.])],
            # Deliberately wrong 'old' values: the custom DIS loss must ignore them.
            "log_probs": [torch.full((3,), -100.), torch.full((1,), -100.)]}
        self.logits = torch.tensor([[[.1, .3, -.2], [.3, -.3, .2], [0., .1, .2], [0., 0., 0.],
                                     [.3, .5, -.5], [0., 0., 0.]]], requires_grad=True)

    def extract_cpu(self, logits, **kwargs):
        scores, entropies, offset = [], [], 0
        for tokens, total, response in zip(self.batch["unconcat_tokens"], self.batch["total_lengths"], self.batch["response_lengths"]):
            start = total - response
            lp = logits[0, offset + start - 1:offset + total - 1].log_softmax(-1)
            scores.append(lp.gather(-1, tokens[-response:, None]).squeeze(-1))
            entropies.append(-(lp.exp() * lp).sum(-1))
            offset += total
        return {"log_probs": scores, "entropy": entropies}

    def invoke(self, *, reducer=None):
        reducer = reducer or self.cp["get_sum_of_sample_mean"](
            self.batch["total_lengths"], self.batch["response_lengths"], self.batch["loss_masks"])
        with patch.object(grpo_dis, "_load_miles_ops", return_value=(self.extract_cpu,
             self.cp["get_local_response_loss_masks"], self.math["compute_approx_kl"])):
            return grpo_dis.loss_function(self.args, self.batch, self.logits, reducer)

    def test_native_reducer_keeps_original_denominator_and_ignores_old_scores(self):
        loss, metrics = self.invoke()
        lp = self.extract_cpu(self.logits)["log_probs"]
        # First sample has two action tokens; its observation and sample two's
        # out-of-region token must not change the denominator of those two.
        ratio0 = (lp[0][0] - self.batch["rollout_log_probs"][0][0]).exp()
        ratio2 = (lp[0][2] - self.batch["rollout_log_probs"][0][2]).exp()
        oracle = (-ratio0 + ratio2) / 2
        torch.testing.assert_close(loss, oracle)
        self.assertEqual(metrics["dis_high_mask_fraction"].item(), 1.)
        # Metrics are sums of per-sample means, as required by Miles logging;
        # the outer dispatcher later averages over the batch.
        self.assertEqual(metrics["dis_kept_fraction"].item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        torch.testing.assert_close(self.logits.grad[0, 1], torch.zeros(3))
        torch.testing.assert_close(self.logits.grad[0, 4], torch.zeros(3))
        self.assertTrue(all(not x.requires_grad for x in metrics.values()))

    def test_rejected_pg_tokens_still_get_independent_kl_regularization(self):
        self.args.use_kl_loss = True
        self.args.kl_loss_coef = .1
        self.batch["ref_log_probs"] = [torch.tensor([-2., float("nan"), -2.]), torch.tensor([-2.])]
        loss, metrics = self.invoke()
        loss.backward()
        self.assertIn("kl_loss", metrics)
        self.assertGreater(self.logits.grad[0, 4].abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_all_rejected_batch_has_zero_pg_gradient_without_nan(self):
        self.batch["rollout_log_probs"] = [torch.full((3,), -1000.), torch.full((1,), -1000.)]
        loss, metrics = self.invoke()
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(metrics["dis_kept_fraction"].item(), 0.)
        torch.testing.assert_close(self.logits.grad, torch.zeros_like(self.logits))

    def test_ppo_clip_parameters_cannot_change_dis(self):
        self.args.eps_clip = .001
        self.args.eps_clip_high = .001
        first, _ = self.invoke()
        self.args.eps_clip = .99
        self.args.eps_clip_high = 20.
        second, _ = self.invoke()
        torch.testing.assert_close(first, second)

    def test_entropy_and_custom_loss_microbatch_scaling_contract(self):
        self.args.entropy_coef = .03
        self.args.loss_type = "custom_loss"
        self.args.custom_loss_function_path = NS(path="algorithms.grpo_dis.loss_function")
        self.args.recompute_loss_function = False
        self.args.true_on_policy_mode = False
        self.args.allgather_cp = False
        self.args.use_dynamic_global_batch_size = False
        objective = native_functions("miles/backends/training_utils/loss/objective.py", ("loss_function",),
            get_parallel_state=lambda: self.parallel, get_sum_of_sample_mean=self.cp["get_sum_of_sample_mean"],
            get_loss_function=lambda *args: grpo_dis.loss_function,
            compute_custom_function_config=lambda *args: self.args, is_multi_lora_enabled=lambda args: False)
        with patch.object(grpo_dis, "_load_miles_ops", return_value=(self.extract_cpu,
             self.cp["get_local_response_loss_masks"], self.math["compute_approx_kl"])):
            direct, _ = self.invoke()
            for megatron in (False, True):
                scaled, normalizer, log = objective["loss_function"](self.args, self.batch, 3, self.logits,
                    apply_megatron_loss_scaling=megatron, num_rollouts=2)
                torch.testing.assert_close(scaled, direct * (3 if megatron else 1) / 2)
                self.assertEqual(normalizer.item(), 1)
                self.assertIn("dis_kept_fraction", log["keys"])

    def test_native_per_token_scaling_keeps_rejected_tokens_in_count(self):
        self.args.backend.calculate_per_token_loss = True
        self.args.loss_type = "custom_loss"
        self.args.custom_loss_function_path = NS(path="algorithms.grpo_dis.loss_function")
        self.args.recompute_loss_function = False
        self.args.true_on_policy_mode = False
        self.args.allgather_cp = False
        self.args.use_dynamic_global_batch_size = False
        objective = native_functions("miles/backends/training_utils/loss/objective.py", ("loss_function",),
            get_parallel_state=lambda: self.parallel, get_sum_of_sample_mean=self.cp["get_sum_of_sample_mean"],
            get_loss_function=lambda *args: grpo_dis.loss_function,
            compute_custom_function_config=lambda *args: self.args, is_multi_lora_enabled=lambda args: False)
        reducer = self.cp["get_sum_of_sample_mean"](self.batch["total_lengths"], self.batch["response_lengths"],
                                                  self.batch["loss_masks"], True)
        raw, _ = self.invoke(reducer=reducer)
        with patch.object(grpo_dis, "_load_miles_ops", return_value=(self.extract_cpu,
             self.cp["get_local_response_loss_masks"], self.math["compute_approx_kl"])):
            scaled, normalizer, _ = objective["loss_function"](self.args, self.batch, 1, self.logits,
                apply_megatron_loss_scaling=False, num_rollouts=2, num_step_tokens=torch.tensor(3.))
        self.assertEqual(normalizer.item(), 3.)  # includes sample two's rejected action
        torch.testing.assert_close(scaled, raw / 3.)

    def test_missing_logprobs_alignment_and_double_correction_are_errors(self):
        original = copy.deepcopy(self.batch)
        del self.batch["rollout_log_probs"]
        with self.assertRaisesRegex(ValueError, "Missing actual"):
            self.invoke()
        self.batch = original
        self.batch["rollout_log_probs"][0] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            self.invoke()
        self.batch = original
        self.args.use_tis = True
        with self.assertRaisesRegex(ValueError, "stacked"):
            self.invoke()

    def test_context_parallel_two_rank_slices_sum_to_unsharded_objective(self):
        full_lp = torch.tensor([-3., -2.5, -2., -1.5, -.5, -2.], dtype=torch.float64)
        full_q = torch.full_like(full_lp, -2.)
        full_a = torch.tensor([1., -1., 2., 3., -2., 1.], dtype=torch.float64)
        mask = torch.tensor([1., 0., 1., 1., 1., 1.], dtype=torch.float64)
        expected = dis_policy_terms(full_lp, full_q, full_a, mask).policy_loss.sum() / mask.sum()
        for layout in ("thd", "bshd"):
            combined = 0
            for rank in (0, 1):
                self.parallel.cp = NS(size=2, rank=rank)
                maximum = [12] if layout == "bshd" else None
                slice_row = lambda row: self.cp["slice_log_prob_with_cp"](row, 12, 6, layout, 12 if maximum else None)
                local_mask = self.cp["get_local_response_loss_masks"]([12], [6], [mask], layout, maximum)[0]
                terms = dis_policy_terms(slice_row(full_lp), slice_row(full_q), slice_row(full_a), local_mask)
                reduce = self.cp["get_sum_of_sample_mean"]([12], [6], [mask], False, layout, maximum)
                combined = combined + reduce(terms.policy_loss)
            torch.testing.assert_close(combined, expected)

    def test_real_function_registry_and_hook_cli_schema(self):
        from miles.utils.function_registry import load_function
        from miles.utils.args.custom_function import CustomFunctionConfig
        parser = argparse.ArgumentParser()
        grpo_dis.DISConfig.add_arguments(parser)
        parsed = parser.parse_args(["--dis-ratio-low", ".3", "--dis-ratio-high", "3."])
        config = grpo_dis.DISConfig.model_validate(vars(parsed))
        selected = CustomFunctionConfig(path="algorithms.grpo_dis.loss_function", config=config)
        self.assertIs(load_function(selected), grpo_dis.loss_function)
        self.assertEqual(selected.config.dis_ratio_low, .3)

    def test_worker_can_import_custom_loss_from_miles_working_directory(self):
        config = read_config(PROJECT / "configs/local.example.json")
        config["workspace_root"] = str(PROJECT)
        environment = {**os.environ, **runtime_env(config), "PYTHONIOENCODING": "utf-8"}
        result = subprocess.check_output([sys.executable, "-c",
            "from miles.utils.function_registry import load_function; "
            "f=load_function('algorithms.grpo_dis.loss_function'); "
            "assert callable(f) and f.config_class.__name__=='DISConfig'; print('hook import OK')"],
            cwd=PROJECT / ".deps/miles", env=environment, encoding="utf-8")
        self.assertIn("hook import OK", result)


if __name__ == "__main__":
    unittest.main()
