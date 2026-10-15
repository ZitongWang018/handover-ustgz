"""CPU-only integration checks for the strict v2 evaluation contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ez_eval_v2
from ez_eval_v2_contract import (SCHEMA, canonical_hash, request_seed, sample_seed, strict_merge,
                                 validate_suite)


def suite_fixture(tmp_path, count=3):
    tasks=[]
    for i in range(count):
        p=tmp_path/f"task{i}"/"game.tw-pddl"; p.parent.mkdir(parents=True); p.write_text(f"game {i}")
        typ="pick_and_place_simple" if i % 2 == 0 else "look_at_obj_in_light"
        tasks.append({"task_id":f"task{i}","task_type":typ,"game_file":str(p.resolve()),
                      "game_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    cfg=tmp_path/"config.yaml"; cfg.write_text("env: {}\n")
    sources={"fake":"source-hash","compatibility":{"mode":"native"}}
    pool=canonical_hash({"split":"valid_unseen","config_sha256":hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "environment_sources":sources,"tasks":[{k:t[k] for k in ("task_id","task_type","game_sha256")} for t in tasks]})
    nsh=min(8,count); q,r=divmod(count,nsh); shards=[]; off=0
    for rank in range(nsh):
        n=q+(rank<r); sub=tasks[off:off+n]
        shards.append({"rank":rank,"offset":off,"task_ids":[t["task_id"] for t in sub],"task_hash":canonical_hash(sub)})
        off+=n
    return {"schema":SCHEMA,"split":"valid_unseen","alfworld_split":"eval_out_of_distribution",
        "config_path":str(cfg.resolve()),"config_sha256":hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "environment_sources":sources,"pool_sha256":pool,"tasks":tasks,"n_tasks":count,
        "n_task_types":len({t["task_type"] for t in tasks}),"requested_shards":8,"shards":shards,
        "sample_seed_policy":"sha256(run_seed,task_id,sample_index)->uint32",
        "samples_per_task":4}


def result_for(contract, rank, *, wrong_temp=False):
    spec=contract["suite"]["shards"][rank]
    sig=dict(contract["protocol_signature"])
    if wrong_temp: sig["decode"]={**sig["decode"],"temperature":0.4}
    lookup={t["task_id"]:t for t in contract["suite"]["tasks"]}
    rows=[]
    for tid in spec["task_ids"]:
        t=lookup[tid]
        for s in range(4):
            row={"task_id":tid,"task_type":t["task_type"],"game_sha256":t["game_sha256"],
                 "game_file":t["game_file"],"sample_index":s,"seed":sample_seed(sig["run_seed"],tid,s),"won":bool(s%2),
                 "steps":1,"actions":["look"],"trace":[{"step":0,"prompt_token_ids":[1,2],
                 "output_token_ids":[3],"raw_output":"look","parser_projection":{"kind":"action","action":"look"},
                 "executed_command":"look","observation":"done","won":bool(s%2),"done":True,
                 "admissible_actions":["look"],"finish_reason":"eos","truncated":False,
                 "stop_token_count":1,"request_seed":request_seed(sample_seed(sig["run_seed"],tid,s),0),
                 "resolver_mode":"v2_recovery_disabled"}],"initial_prompt_token_ids":[1,2],"chat_template":"fake"}
            row["record_sha256"]=canonical_hash(row); rows.append(row)
    return {"schema":SCHEMA,"run_id":contract["run_id"],"contract_sha256":contract["contract_sha256"],
        "rank":rank,"split":contract["suite"]["split"],"pool_sha256":contract["suite"]["pool_sha256"],
        "task_ids":spec["task_ids"],"protocol_signature":sig,"arm_identity":contract["arm_identity"],
        "base_inventory":contract["base_inventory"],"software":contract["software"],
        "runtime_source_sha256":contract["runtime_source_sha256"],"sample_rows":rows}


def seal_contract(c):
    c["contract_sha256"]=canonical_hash(c)
    return c


def signature(suite, seed=17):
    return {"pool_sha256":suite["pool_sha256"],"split":suite["split"],"samples_per_task":4,
        "protocol":"paper-approx","prompt_sha256":"p","tokenizer_sha256":"t","scaffold_sha256":"p",
        "chat_template_sha256":"c","parser":"ez_json_tool_call/v2","menu":False,"stop":"tool_call",
        "decode":{"temperature":1.0,"top_p":0.95,"top_k":-1,"repetition_penalty":1.0},
        "max_context_tokens":40960,"max_new_tokens":1024,"max_steps":30,"seed_policy":"sha256",
        "run_seed":seed,"backend":"hf","base_inventory_sha256":"base","attention_backend":"sdpa",
        "dtype":"bfloat16","software":{"python":"test"},"runtime_source_sha256":{"eval":"hash"},
        "resolver":"no_hidden_admissible_recovery; malformed_projects_to_look","fewshot":False,"parser_strict":False}


def test_suite_dynamic_partition_and_bad_content_rejected(tmp_path):
    s=suite_fixture(tmp_path,3)
    validate_suite(s)
    assert s["n_tasks"] != 134 and len(s["shards"]) == 3
    assert all(x["task_ids"] for x in s["shards"])
    s["tasks"][0]["task_type"]="tampered"
    with pytest.raises(ValueError,match="content hash"):
        validate_suite(s)
    s=suite_fixture(tmp_path/"empty",1); s["shards"][0]["task_ids"]=[]
    with pytest.raises(ValueError,match="partition"):
        validate_suite(s)


def test_discover_suite_locks_configured_logic_and_trajectory_bytes(tmp_path, monkeypatch):
    domain=tmp_path/"domain.pddl"; domain.write_text("domain-v1")
    grammar=tmp_path/"grammar.tw"; grammar.write_text("grammar-v1")
    game=tmp_path/"task"/"game.tw-pddl"; game.parent.mkdir(parents=True); game.write_text("game")
    traj=game.parent/"traj_data.json"; traj.write_text('{"traj":1}')
    config=tmp_path/"alfworld.yaml"; config.write_text("logic: {domain: domain.pddl, grammar: grammar.tw}\n")
    env_mod=types.ModuleType("alfworld_env")
    env_mod.collect_game_files=lambda cfg,split:[str(game)]
    env_mod.load_alfworld_config=lambda path:{"logic":{"domain":str(domain),"grammar":str(grammar)}}
    env_mod.task_instance_id=lambda path:"task-1"
    env_mod.task_type_of=lambda path:"pick_and_place_simple"
    env_mod.ALFWorldBatchEnv=type("FakeALFEnv",(),{"__module__":__name__})
    monkeypatch.setitem(sys.modules,"alfworld_env",env_mod)
    monkeypatch.setattr("ez_eval_v2_contract.ensure_environment_compat",lambda:{"mode":"native"})
    parent=types.ModuleType("alfworld"); parent.__path__=[]
    agents=types.ModuleType("alfworld.agents"); agents.__path__=[]
    environment=types.ModuleType("alfworld.agents.environment"); environment.__path__=[]
    leaf=types.ModuleType("alfworld.agents.environment.alfred_tw_env")
    leaf.AlfredTWEnv=type("FakeTWEnv",(),{"__module__":__name__})
    monkeypatch.setitem(sys.modules,"alfworld",parent)
    monkeypatch.setitem(sys.modules,"alfworld.agents",agents)
    monkeypatch.setitem(sys.modules,"alfworld.agents.environment",environment)
    monkeypatch.setitem(sys.modules,"alfworld.agents.environment.alfred_tw_env",leaf)
    tw=types.ModuleType("textworld"); tw.__version__="fake"
    monkeypatch.setitem(sys.modules,"textworld",tw)
    import ez_eval_v2_contract as contract
    first=contract.discover_suite(str(config),"valid_unseen")
    grammar.write_text("grammar-v2")
    second=contract.discover_suite(str(config),"valid_unseen")
    assert first["pool_sha256"]!=second["pool_sha256"]
    assert first["config_sha256"]==second["config_sha256"]
    assert first["environment_sources"]["environment_files"]["logic_files"]["grammar"]["sha256"]!=second["environment_sources"]["environment_files"]["logic_files"]["grammar"]["sha256"]
    traj.write_text('{"traj":2}')
    third=contract.discover_suite(str(config),"valid_unseen")
    assert second["pool_sha256"]!=third["pool_sha256"]


def test_strict_merge_requires_predeclared_contract_and_full_grid(tmp_path):
    suite=suite_fixture(tmp_path,3)
    sig=signature(suite)
    c={"schema":SCHEMA,"run_id":"run-a","suite":suite,"protocol_signature":sig,
       "arm_identity":{"kind":"base-only","base_inventory_sha256":"base"},"base_inventory":[],"software":{"python":"test"},"runtime_source_sha256":{"eval":"hash"},
       "expected_shards":[{"rank":s["rank"],"task_ids":s["task_ids"],"path":f"rank{s['rank']}.json"} for s in suite["shards"]]}
    seal_contract(c)
    merged=strict_merge(c,[result_for(c,i) for i in range(3)])
    assert merged["summary"]["n_tasks"]==3 and merged["summary"]["n_episodes"]==12
    assert merged["summary"]["six_type_macro_sr"] is None
    assert merged["summary"]["task_type_sr"]["pick_clean_then_place_in_recep"] is None
    with pytest.raises(ValueError,match="protocol signature"):
        strict_merge(c,[result_for(c,0,wrong_temp=True),result_for(c,1,wrong_temp=True),result_for(c,2,wrong_temp=True)])
    missing=result_for(c,0); missing["sample_rows"].pop()
    with pytest.raises(ValueError,match="sample grid"):
        strict_merge(c,[missing,*[result_for(c,i) for i in (1,2)]])
    corrupt=result_for(c,0); corrupt["sample_rows"][0]["won"]=not corrupt["sample_rows"][0]["won"]
    with pytest.raises(ValueError,match="content hash"):
        strict_merge(c,[corrupt,*[result_for(c,i) for i in (1,2)]])
    inconsistent=result_for(c,0); inconsistent["sample_rows"][0]["won"]=not inconsistent["sample_rows"][0]["won"]
    row=dict(inconsistent["sample_rows"][0]); row.pop("record_sha256"); inconsistent["sample_rows"][0]["record_sha256"]=canonical_hash(row)
    with pytest.raises(ValueError,match="final trace"):
        strict_merge(c,[inconsistent,*[result_for(c,i) for i in (1,2)]])
    bad_action=result_for(c,0); bad_action["sample_rows"][0]["actions"][0]="open cabinet"
    row=dict(bad_action["sample_rows"][0]); row.pop("record_sha256"); bad_action["sample_rows"][0]["record_sha256"]=canonical_hash(row)
    with pytest.raises(ValueError,match="action list"):
        strict_merge(c,[bad_action,*[result_for(c,i) for i in (1,2)]])
    bad_prompt=result_for(c,0); bad_prompt["sample_rows"][0]["initial_prompt_token_ids"]=[9]
    row=dict(bad_prompt["sample_rows"][0]); row.pop("record_sha256"); bad_prompt["sample_rows"][0]["record_sha256"]=canonical_hash(row)
    with pytest.raises(ValueError,match="first trace prompt"):
        strict_merge(c,[bad_prompt,*[result_for(c,i) for i in (1,2)]])


def test_legacy_checkpoint_is_rejected_before_any_model_load(tmp_path):
    base=tmp_path/"base"; base.mkdir(); (base/"config.json").write_text("{}")
    (base/"weights.safetensors").write_bytes(b"not loaded in CPU preflight")
    legacy=tmp_path/"old-checkpoint"; legacy.mkdir(); (legacy/"solver").mkdir()
    (legacy/"solver"/"adapter_config.json").write_text("{}")
    from ez_train_v2_contract import base_inventory_sha256
    args=argparse.Namespace(base_only=False,checkpoint=str(legacy))
    with pytest.raises(ValueError,match="missing required"):
        ez_eval_v2._checkpoint_info(args,base_inventory_sha256(base))


def test_compare_pair_rejects_protocol_or_seed_mismatch(tmp_path):
    sig=signature(suite_fixture(tmp_path,1),2); sig["backend"]="vllm"; sig["attention_backend"]="vllm"
    def rows(won):
        out=[]
        for i in range(4):
            row={"task_id":"x","task_type":"pick_and_place_simple","game_file":"game.tw-pddl","game_sha256":"gh",
                 "sample_index":i,"seed":sample_seed(2,"x",i),"won":won}
            row["record_sha256"]=canonical_hash(row); out.append(row)
        return out
    base={"schema":SCHEMA,"protocol_signature":dict(sig),"arm_identity":{"kind":"base-only","base_inventory_sha256":"base"},
        "expected_task_ids":["x"],"summary":{"n_tasks":1,"n_episodes":4,"episode_weighted_sr":0},"sample_rows":rows(False)}
    adapter={"schema":SCHEMA,"protocol_signature":dict(sig),"arm_identity":{"kind":"adapter","base_inventory_sha256":"base"},
        "expected_task_ids":["x"],"summary":{"n_tasks":1,"n_episodes":4,"episode_weighted_sr":1},"sample_rows":rows(True)}
    assert ez_eval_v2.compare_cmd
    from ez_eval_v2_contract import compare_pair
    assert compare_pair(base,adapter)["n_pairs"]==4
    adapter["protocol_signature"]["max_steps"]=50
    with pytest.raises(ValueError,match="max_steps"):
        compare_pair(base,adapter)
    adapter["protocol_signature"]["max_steps"]=30
    adapter["sample_rows"][0]["seed"]+=1
    row=dict(adapter["sample_rows"][0]); row.pop("record_sha256")
    adapter["sample_rows"][0]["record_sha256"]=canonical_hash(row)
    with pytest.raises(ValueError,match="sample seed"):
        compare_pair(base,adapter)
    adapter["protocol_signature"].pop("runtime_source_sha256")
    with pytest.raises(ValueError,match="required evaluation fields"):
        compare_pair(base,adapter)
    adapter["protocol_signature"]=dict(sig)
    adapter["arm_identity"]["base_inventory_sha256"]="different-base"
    with pytest.raises(ValueError,match="base digest"):
        compare_pair(base,adapter)


def test_compare_pair_rejects_missing_schema_or_base_identity_digest(tmp_path):
    from ez_eval_v2_contract import compare_pair
    with pytest.raises(ValueError,match="merged ez-eval-v2"):
        compare_pair({}, {})
    sig=signature(suite_fixture(tmp_path,2),3)
    def make_rows():
        rows=[]
        for i in range(4):
            row={"task_id":"x","task_type":"pick_and_place_simple","game_file":"x.tw-pddl",
                "game_sha256":"h","sample_index":i,"seed":sample_seed(3,"x",i),"won":False}
            row["record_sha256"]=canonical_hash(row); rows.append(row)
        return rows
    a={"schema":SCHEMA,"protocol_signature":sig,"arm_identity":{"kind":"base-only","base_inventory_sha256":"base"},
        "expected_task_ids":["x","y"],"summary":{"n_tasks":1,"n_episodes":4,"episode_weighted_sr":0},"sample_rows":make_rows()}
    b={"schema":SCHEMA,"protocol_signature":sig,"arm_identity":{"kind":"adapter","base_inventory_sha256":"base"},
        "expected_task_ids":["x","y"],"summary":{"n_tasks":1,"n_episodes":4,"episode_weighted_sr":0},"sample_rows":make_rows()}
    with pytest.raises(ValueError,match="full expected task manifest"):
        compare_pair(a,b)


@pytest.mark.parametrize("field,value",[("temperature",float("nan")),("top_p",float("inf"))])
def test_prepare_run_rejects_nonfinite_sampling_before_io(field,value):
    opts=argparse.Namespace(batch=1,max_steps=1,max_new_tokens=1,max_context_tokens=2,
        temperature=1.0,top_p=.9,top_k=-1,gpu_memory_utilization=.8)
    setattr(opts,field,value)
    with pytest.raises(ValueError,match="invalid sampling settings"):
        ez_eval_v2.prepare_run(opts)


def test_seed_policy_changes_each_turn_but_is_replayable():
    s=sample_seed(42,"task-X",3)
    assert s==sample_seed(42,"task-X",3)
    assert request_seed(s,0)==request_seed(s,0)
    assert request_seed(s,0)!=request_seed(s,1)


def test_controls_generator_guards_gpu_map_and_uses_lf_skr_python(tmp_path):
    sys.path.insert(0,str(ROOT/"scripts"))
    import ez_eval_v2_controls
    suite=suite_fixture(tmp_path/"suite",3)
    manifest=tmp_path/"suite.json"; manifest.write_text(json.dumps(suite),encoding="utf-8")
    out=tmp_path/"commands"
    ez_eval_v2_controls.generate(suite_manifest=manifest,base="/local/base",checkpoint="/local/v3",
        config=suite["config_path"],out_dir=out,backend="vllm",seed=11)
    script=(out/"run.sh").read_bytes()
    assert b"\r" not in script
    text=script.decode("utf-8")
    assert 'PYTHON_BIN="${SKR_PYTHON:-python}"' in text
    assert 'CUDA_VISIBLE_DEVICES contains an empty GPU ID' in text
    assert 'duplicate CUDA_VISIBLE_DEVICES entry' in text
    assert 'CUDA_VISIBLE_DEVICES="${GPU_MAP[0]}" "$PYTHON_BIN"' in text
    assert 'if ((FAILED)); then echo "shard failed; skipping merge"' in text
    portable=(ROOT/"scripts"/"run_ez_eval_v2_8gpu.sh").read_text(encoding="utf-8")
    assert 'PYTHON_BIN="${SKR_PYTHON:-python}"' in portable
    assert 'duplicate CUDA_VISIBLE_DEVICES entry' in portable


def test_hf_policy_native_seeded_generation_and_greedy_tie(tmp_path):
    torch=pytest.importorskip("torch")
    transformers=pytest.importorskip("transformers")
    sys.path.insert(0,str(ROOT/"src"))
    from rollout import HFPolicy, RolloutConfig
    class Tok:
        pad_token_id=9; eos_token_id=10; eos_token="<eos>"; padding_side="right"; chat_template="tiny"
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs): return "prompt"
        def __call__(self, text, add_special_tokens=False): return {"input_ids":[1,2]}
        def pad(self, data, padding=True, return_tensors="pt"):
            ids=data["input_ids"]; width=max(map(len,ids)); padded=[]; masks=[]
            for row in ids:
                n=width-len(row); padded.append([self.pad_token_id]*n+list(row)); masks.append([0]*n+[1]*len(row))
            return transformers.tokenization_utils_base.BatchEncoding({
                "input_ids":torch.tensor(padded,dtype=torch.long),"attention_mask":torch.tensor(masks,dtype=torch.long)})
        def decode(self, ids, skip_special_tokens=True): return " ".join(map(str,ids))
    config=transformers.GPT2Config(vocab_size=11,n_positions=16,n_ctx=16,n_embd=8,
        n_layer=1,n_head=1,bos_token_id=1,eos_token_id=10,pad_token_id=9)
    model=transformers.GPT2LMHeadModel(config).eval()
    tok=Tok(); policy=HFPolicy(model,tok,device="cpu")
    cfg=RolloutConfig(max_new_tokens=3,temperature=1.0,top_p=1.0,top_k=-1,
        repetition_penalty=1.0,max_context_tokens=16,stop_mode="eos")
    prompts=[[1,2],[1,3,4]]; seeds=[123,456]
    first=policy.generate_from_token_ids(prompts,cfg,request_seeds=seeds)
    second=policy.generate_from_token_ids(prompts,cfg,request_seeds=seeds)
    assert first[1]==second[1] and len(first[1])==2
    with torch.no_grad():
        for param in model.parameters(): param.zero_()
    greedy=RolloutConfig(max_new_tokens=2,temperature=0.0,top_p=1.0,top_k=-1,
        repetition_penalty=1.0,max_context_tokens=16,stop_mode="eos")
    seeded=policy.generate_from_token_ids([[1,2]],greedy,request_seeds=[999])[1][0]
    native=model.generate(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones((1,2),dtype=torch.long),
        max_new_tokens=2,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    expected=[x for x in native[0,2:].tolist() if x != tok.pad_token_id]
    assert seeded==expected==[0,0]
    import eval_ez
    from types import SimpleNamespace
    class Env:
        def reset(self,games):
            views=[SimpleNamespace(env_desc="room",raw="room",task_desc="find apple") for _ in games]
            return ["room"]*len(games),views,[["look"] for _ in games],list(games),[None]*len(games)
        def step(self,actions):
            return SimpleNamespace(won=[False]*len(actions),admissible=[["look"] for _ in actions],
                obs=["nothing happens"]*len(actions),dones=[True]*len(actions))
    traced=eval_ez._rollout_batch(Env(),policy,["/fake/a/game.tw-pddl","/fake/b/game.tw-pddl"],
        [{"task_id":"a","task_type":"pick_and_place_simple","sample":0,"sample_seed":101},
         {"task_id":"b","task_type":"look_at_obj_in_light","sample":0,"sample_seed":202}],
        RolloutConfig(max_steps=1,max_new_tokens=3,temperature=1.0,top_p=1.0,top_k=-1,
            repetition_penalty=1.0,max_context_tokens=16,stop_mode="tool_call"),1,
        style="ours",menu_enabled=False,capture_v2_trace=True)
    assert len(traced)==2 and all(len(x["trace"])==1 for x in traced)
    assert len(policy.last_finish_reasons)==2


def test_fake_cpu_run_prepares_then_executes_and_merges_contract(tmp_path, monkeypatch):
    suite=suite_fixture(tmp_path,3)
    monkeypatch.setattr(ez_eval_v2,"discover_suite",lambda *a,**k:suite)
    class Tok:
        pad_token_id=0; eos_token="<eos>"; chat_template="fake-chat-template"
    transformers=types.ModuleType("transformers")
    transformers.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok())
    monkeypatch.setitem(sys.modules,"transformers",transformers)
    model=tmp_path/"model"; model.mkdir(); (model/"config.json").write_text("{}")
    (model/"weights.safetensors").write_bytes(b"cpu-fake-weight")
    results=tmp_path/"results"
    opts=argparse.Namespace(suite_manifest=str(tmp_path/"suite.json"),config=suite["config_path"],
        base=str(model),checkpoint=None,base_only=True,backend="hf",seed=9,run_id="fake-run",
        results_dir=str(results),output=str(tmp_path/"contract.json"),batch=2,max_steps=30,
        max_context_tokens=40960,max_new_tokens=1024,temperature=1.0,top_p=.95,top_k=-1,
        gpu_memory_utilization=.85,attn="sdpa")
    Path(opts.suite_manifest).write_text(json.dumps(suite),encoding="utf-8")
    ez_eval_v2.prepare_run(opts)
    class Env:
        def __init__(self,*a,**k): pass
        def close(self): pass
    alf=types.ModuleType("alfworld_env"); alf.ALFWorldBatchEnv=Env; alf.load_alfworld_config=lambda p:{}
    monkeypatch.setitem(sys.modules,"alfworld_env",alf)
    def fake_rollout(env,policy,game_files,records,cfg,max_steps,**kwargs):
        return [{"task_id":r["task_id"],"task_type":r["task_type"],"game_file":g,
            "sample":r["sample"],"sample_seed":r["sample_seed"],"won":bool(r["sample"]%2),
            "steps":1,"trace":[{"step":0,"prompt_token_ids":[1,2],"output_token_ids":[3],
                "raw_output":"look","parser_projection":{"action":"look"},"executed_command":"look",
                "observation":"room","won":bool(r["sample"]%2),"done":True,"admissible_actions":["look"],
                "finish_reason":"eos","truncated":False,"stop_token_count":1,
                "request_seed":request_seed(r["sample_seed"],0),"resolver_mode":"v2_recovery_disabled"}],
            "initial_prompt_token_ids":[1,2],"chat_template":"fake-chat-template",
            "actions":["look"],"kinds":{}} for g,r in zip(game_files,records)]
    fake_eval=types.ModuleType("eval_ez"); fake_eval._rollout_batch=fake_rollout
    monkeypatch.setitem(sys.modules,"eval_ez",fake_eval)
    monkeypatch.setattr(ez_eval_v2,"_load_v2_policy",lambda args,path,tok=None:types.SimpleNamespace(tokenizer=tok))
    contract=json.loads(Path(opts.output).read_text(encoding="utf-8"))
    shard_results=[]
    for sh in suite["shards"]:
        dest=Path(contract["expected_shards"][sh["rank"]]["path"])
        ez_eval_v2.run_eval(argparse.Namespace(run_contract=opts.output,rank=sh["rank"],
            world_size=len(suite["shards"]),output=str(dest)))
        shard_results.append(json.loads(dest.read_text(encoding="utf-8")))
    merged=strict_merge(contract,shard_results)
    assert merged["summary"]["n_episodes"]==12
    assert all(len(row["initial_prompt_token_ids"])==2 for row in merged["sample_rows"])


def test_real_rollout_parser_binds_projection_to_executed_action(monkeypatch):
    from types import SimpleNamespace
    import eval_ez
    game="/fake/task/game.tw-pddl"
    class Env:
        def __init__(self, raw): self.raw=raw; self.actions=[]
        def reset(self, games):
            view=SimpleNamespace(env_desc="room",raw="room",task_desc="find apple")
            return ["room"],[view],[["go to cabinet 1","look"]],[game],[None]
        def step(self, actions):
            self.actions.extend(actions)
            return SimpleNamespace(won=[False],admissible=[["look"]],obs=["nothing happens"],dones=[True])
    class Policy:
        tokenizer=SimpleNamespace(chat_template="fake")
        def __init__(self, text): self.text=text
        def initial_prompt_ids(self,*a): return [1,2]
        def generate_from_token_ids(self,prompts,cfg,*,request_seeds=None):
            if request_seeds is not None: assert len(request_seeds)==1
            return [self.text],[[3,4]],[2]
    rec=[{"task_id":"task","task_type":"pick_and_place_simple","sample":0,"sample_seed":123}]
    cfg=SimpleNamespace(max_context_tokens=64,max_new_tokens=8)
    legacy_env=Env("legacy"); legacy=eval_ez._rollout_batch(legacy_env,Policy("go to cabinet 1"),[game],rec,cfg,1,style="ours",menu_enabled=False)
    assert legacy_env.actions==["go to cabinet 1"]
    v2_env=Env("v2"); v2=eval_ez._rollout_batch(v2_env,Policy("go to cabinet 1"),[game],rec,cfg,1,style="ours",menu_enabled=False,capture_v2_trace=True)
    assert v2_env.actions==["look"] and v2[0]["trace"][0]["resolver_mode"]=="v2_recovery_disabled"
    json_env=Env("json"); parsed=eval_ez._rollout_batch(json_env,Policy('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"cabinet 1"}}</tool_call>'),[game],rec,cfg,1,style="ours",menu_enabled=False,capture_v2_trace=True)
    assert json_env.actions==["go to cabinet 1"]
    trace=parsed[0]["trace"][0]
    assert trace["parser_projection"]["action"]==trace["executed_command"]
    assert trace["prompt_token_ids"]==[1,2] and trace["output_token_ids"]==[3,4]
