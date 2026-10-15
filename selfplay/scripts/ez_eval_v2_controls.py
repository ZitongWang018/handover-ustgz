#!/usr/bin/env python3
"""Generate reviewable paired v2 evaluation commands; never executes them."""
from __future__ import annotations
import argparse, json, shlex, uuid
from pathlib import Path

def generate(*, suite_manifest, base, checkpoint, config, out_dir, backend="vllm", seed=42):
    suite_path = Path(suite_manifest).resolve(strict=True)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema") != "ez-eval-v2" or not suite.get("shards"):
        raise ValueError("expected a prepared non-empty v2 suite manifest")
    out = Path(out_dir).resolve()
    if out.exists(): raise FileExistsError(f"refusing to overwrite output directory: {out}")
    out.mkdir(parents=True)
    run = uuid.uuid4().hex
    jobs=[]
    for arm, ckpt in (("base", None), ("adapter", checkpoint)):
        if arm == "adapter" and not ckpt: continue
        contract = out / f"{arm}-contract.json"
        results_dir = out / arm
        prepare = ["python", "selfplay/src/ez_eval_v2.py", "prepare-run", "--suite-manifest", str(suite_path),
            "--config", str(Path(config).resolve()), "--base", base, "--backend", backend,
            "--seed", str(seed), "--run-id", run, "--results-dir", str(results_dir), "--output", str(contract)]
        if ckpt is None: prepare += ["--base-only"]
        else: prepare += ["--checkpoint", ckpt]
        jobs.append({"arm": arm, "phase": "prepare", "argv": prepare, "shell": shlex.join(prepare)})
        for shard in suite["shards"]:
            dest = results_dir / f"rank{shard['rank']}.json"
            argv = ["python", "selfplay/src/ez_eval_v2.py", "run", "--run-contract", str(contract),
                    "--rank", str(shard["rank"]), "--world-size", str(len(suite["shards"])), "--output", str(dest)]
            jobs.append({"arm": arm, "phase": "run", "rank": shard["rank"], "argv": argv,
                         "shell": shlex.join(argv), "output": str(dest)})
        merge = ["python", "selfplay/src/ez_eval_v2.py", "merge", "--expected-manifest", str(contract),
                 "--output", str(out / f"{arm}-merged.json")]
        jobs.append({"arm": arm, "phase": "merge", "argv": merge, "shell": shlex.join(merge)})
    if checkpoint:
        pair=["python","selfplay/src/ez_eval_v2.py","compare-pair","--base-result",str(out/"base-merged.json"),
              "--adapter-result",str(out/"adapter-merged.json"),"--output",str(out/"paired-difference.json")]
        jobs.append({"phase":"compare-pair","argv":pair,"shell":shlex.join(pair)})
    plan={"schema":"ez-eval-v2-plan", "execution_policy":"manual_only", "suite_manifest":str(suite_path),
          "run_id":run, "expected_tasks":suite["n_tasks"], "nonempty_shards":len(suite["shards"]),
          "paired_protocol":"base-only vs v3 solver adapter; run separately with matched suite/seed/decode",
          "jobs":jobs}
    (out/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    lines=["#!/usr/bin/env bash","set -euo pipefail","# Review and run these commands manually.",
           f"cd {shlex.quote(str(repo_root))}",
           'PYTHON_BIN="${SKR_PYTHON:-python}"',
           f"N_SHARDS={len(suite['shards'])}",
           'VISIBLE_GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"',
           'case ",${VISIBLE_GPU_IDS}," in *,,*) echo "CUDA_VISIBLE_DEVICES contains an empty GPU ID" >&2; exit 2;; esac',
           'IFS=, read -r -a GPU_MAP <<< "$VISIBLE_GPU_IDS"',
           'if ((${#GPU_MAP[@]} < N_SHARDS)); then echo "not enough visible GPU IDs" >&2; exit 2; fi',
           'for ((i=0; i<${#GPU_MAP[@]}; i++)); do for ((j=0; j<i; j++)); do if [[ "${GPU_MAP[$i]}" == "${GPU_MAP[$j]}" ]]; then echo "duplicate CUDA_VISIBLE_DEVICES entry: ${GPU_MAP[$i]}" >&2; exit 2; fi; done; done']
    def shell_python(argv):
        return '"$PYTHON_BIN" ' + shlex.join(argv[1:])
    for arm, ckpt in (("base", None), ("adapter", checkpoint)):
        if arm == "adapter" and not ckpt: continue
        contract=out/f"{arm}-contract.json"; results_dir=out/arm
        prepare=["python","selfplay/src/ez_eval_v2.py","prepare-run","--suite-manifest",str(suite_path),
            "--config",str(Path(config).resolve()),"--base",base,"--backend",backend,"--seed",str(seed),
            "--run-id",run,"--results-dir",str(results_dir),"--output",str(contract)]
        prepare += ["--base-only"] if ckpt is None else ["--checkpoint",ckpt]
        lines.append(shell_python(prepare))
        lines += ["PIDS=()", "FAILED=0", "cleanup() { for p in \"${PIDS[@]:-}\"; do kill \"$p\" 2>/dev/null || true; done; wait 2>/dev/null || true; }", "trap cleanup INT TERM"]
        for sh in suite["shards"]:
            rank=sh["rank"]
            command=["python","selfplay/src/ez_eval_v2.py","run","--run-contract",str(contract),
                "--rank",str(rank),"--world-size",str(len(suite["shards"])),
                "--output",str(results_dir/f"rank{rank}.json")]
            log_path=out/f"{arm}-rank{rank}.log"
            lines.append(f"CUDA_VISIBLE_DEVICES=\"${{GPU_MAP[{rank}]}}\" {shell_python(command)} >{shlex.quote(str(log_path))} 2>&1 &")
            lines.append('PIDS+=("$!")')
        lines += ['for pid in "${PIDS[@]}"; do if ! wait "$pid"; then FAILED=1; fi; done',"trap - INT TERM",
                  'if ((FAILED)); then echo "shard failed; skipping merge" >&2; exit 1; fi']
        merge=["python","selfplay/src/ez_eval_v2.py","merge","--expected-manifest",str(contract),
               "--output",str(out/f"{arm}-merged.json")]
        lines.append(shell_python(merge))
    if checkpoint:
        pair=["python","selfplay/src/ez_eval_v2.py","compare-pair","--base-result",str(out/"base-merged.json"),
              "--adapter-result",str(out/"adapter-merged.json"),"--output",str(out/"paired-difference.json")]
        lines.append(shell_python(pair))
    script="\n".join(lines)+"\n"
    (out/"run.sh").write_bytes(script.encode("utf-8"))
    return plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-manifest",required=True); p.add_argument("--base",required=True)
    p.add_argument("--checkpoint"); p.add_argument("--config",required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--backend",choices=["hf","vllm"],default="vllm")
    p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    try: generate(suite_manifest=a.suite_manifest,base=a.base,checkpoint=a.checkpoint,config=a.config,out_dir=a.out_dir,backend=a.backend,seed=a.seed)
    except (ValueError,FileExistsError,OSError) as e: p.error(str(e))
    print("wrote manual-only v2 command plan")
if __name__=="__main__": main()
