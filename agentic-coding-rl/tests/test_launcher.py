"""CPU checks for scientific controls and the pinned upstream interfaces."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import launcher
import run


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.config = launcher.read_config(PROJECT / "configs/local.example.json")
        self.config["workspace_root"] = str(PROJECT)
        self.config["runtime"]["session_external_host"] = "192.0.2.10"
        self.run_dir = PROJECT / "runs" / "cpu-test-not-created"

    def command(self, algo="grpo", mode="async", **kwargs):
        return launcher.build_train(self.config, algo, mode, self.run_dir, **kwargs)

    def value(self, command, key):
        return command[command.index("--" + key) + 1]

    def test_equal_trajectory_budget_across_algorithms_and_schedules(self):
        for algo in launcher.ALGORITHMS:
            for mode in ("sync", "async"):
                with self.subTest(algo=algo, mode=mode):
                    c = self.command(algo, mode)
                    self.assertEqual(int(self.value(c, "rollout-batch-size")) * int(self.value(c, "n-samples-per-prompt")), 16)
                    self.assertEqual(self.value(c, "global-batch-size"), "16")
                    self.assertEqual(self.value(c, "actor-num-gpus-per-node"), "4" if mode == "async" else "8")
                    self.assertEqual("--fully-async" in c, mode == "async")
                    self.assertEqual("--colocate" in c, mode == "sync")

    def test_single_rollout_uses_critic_not_degenerate_grpo(self):
        c = self.command("ppo_single")
        self.assertEqual(self.value(c, "advantage-estimator"), "ppo")
        self.assertEqual(self.value(c, "n-samples-per-prompt"), "1")
        self.assertIn("--critic-lr", c)
        self.assertIn("--use-rollout-logprobs", c)
        self.assertEqual(self.value(c, "kl-coef"), "0")
        c = self.command("ppo_single", smoke=True)
        self.assertGreater(int(self.value(c, "num-rollout")), self.config["training"]["critic_warmup_rollouts"])
        with self.assertRaises(ValueError):
            self.command("sao")

    def test_correction_is_not_applied_twice(self):
        for algo, mode in (("grpo_tis", "truncate"), ("grpo_mis", "mask")):
            c = self.command(algo)
            self.assertIn("--use-tis", c)
            self.assertNotIn("--use-rollout-logprobs", c)
            self.assertEqual(self.value(c, "tis-mode"), mode)
        c = self.command("score_centering")
        self.assertNotIn("--use-tis", c)
        self.assertEqual(self.value(c, "loss-type"), "score_centering")
        self.assertIn("--disable-grpo-std-normalization", c)

    def test_bad_budget_topology_and_context_fail_early(self):
        for field, key, value in (("training", "total_trajectories_per_update", 15),
                                  ("runtime", "async_actor_gpus", 3),
                                  ("agent", "max_concurrent", 4),
                                  ("agent", "client_timeout_seconds", 1)):
            c = copy.deepcopy(self.config)
            c[field][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.build_train(c, "grpo", "async", self.run_dir)

    def test_upstream_entrypoints_and_registered_algorithm_flags(self):
        launcher.verify_sources()
        source = launcher.checkout("miles")
        for entry in ("train.py", "train_async.py", "examples/swe-agent-harbor-docker/generate.py",
                      "examples/swe-agent-harbor-docker/swe_agent_function.py"):
            self.assertTrue((source / entry).is_file())
        algo_tree = ast.parse((source / "miles/utils/args/configs/algo.py").read_text(encoding="utf-8"))
        annotated = {node.target.id for node in ast.walk(algo_tree)
                     if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        for name in ("advantage_estimator", "critic_lr", "use_rollout_logprobs", "score_centering_is",
                     "loss_type", "custom_tis_function_path", "num_critic_only_steps"):
            self.assertIn(name, annotated)
        math = (source / "miles/backends/training_utils/loss/hub/math_utils.py").read_text(encoding="utf-8")
        self.assertIn("GAE runs on the", math)
        self.assertIn("subsequence of trainable tokens", math)
        # Load the real pure-Python architecture resolver, not a fake fixture.
        model = launcher.model_args("qwen3-4B-Instruct-2507")
        self.assertEqual(model[model.index("--rotary-base") + 1], "5000000")

    def test_dataset_disjointness_and_no_gold_in_rollout(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as td:
            root = Path(td)
            self.config["workspace_root"] = str(root)
            self.config["paths"].update({"train_jsonl": "train.jsonl", "eval_jsonl": "eval.jsonl", "tasks_dir": "tasks"})
            task = root / "tasks" / "django__django-1"
            for file in ("task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh"):
                path = task / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            row = {"prompt": "fix bug", "metadata": {"instance_id": task.name, "agent_name": "mini-swe-agent", "max_seq_len": 16384}}
            for file in ("train.jsonl", "eval.jsonl"):
                (root / file).write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(run.check_data(self.config), {"train": 1})
            with self.assertRaisesRegex(ValueError, "overlap"):
                run.check_data(self.config, evaluate=True)
            row["metadata"]["patch"] = "gold patch"
            (root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Gold"):
                run.check_data(self.config)

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            launcher.workspace_path(self.config, "../outside-run")

    def test_eval_is_explicit_and_gspo_clip_is_separate(self):
        self.assertNotIn("--eval-prompt-data", self.command())
        c = self.command(evaluate=True)
        self.assertIn("--eval-prompt-data", c)
        self.assertEqual(self.value(c, "n-samples-per-eval-prompt"), "1")
        self.assertNotEqual(self.value(self.command("gspo"), "eps-clip"), self.value(self.command(), "eps-clip"))


if __name__ == "__main__":
    unittest.main()
