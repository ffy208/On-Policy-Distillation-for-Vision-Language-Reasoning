"""Naming and command construction for the experiment runner (no GPU, no network)."""

import pytest

pytest.importorskip("yaml")

from vlm_opd.experiment import (
    TASKS_FILE,
    eval_command,
    load_tasks,
    point_names,
    run_point,
    train_command,
)


def test_legacy_names_match_notebook_outputs():
    cfg = load_tasks(TASKS_FILE)
    d, t = cfg["defaults"], cfg["tasks"]["chartqa_human"]
    n = point_names("chartqa_human", t, d, "opd", 300, 42)
    assert n.result_name == "eval_opd_q300.json"
    assert n.merged_repo == "ffyang/vlm_opd_opd_human_q300_merged"
    assert n.ckpt_repo == "ffyang/vlm_opd_opd_human_q300_ckpt"
    s = point_names("chartqa_human", t, d, "sft", 3000, 42)
    assert s.result_name == "eval_sft_q3000.json" and s.ckpt_repo is None


def test_seeded_names_are_distinct_and_task_scoped():
    cfg = load_tasks(TASKS_FILE)
    d, t = cfg["defaults"], cfg["tasks"]["chartqa_human"]
    a = point_names("chartqa_human", t, d, "opd", 300, 1)
    b = point_names("chartqa_human", t, d, "opd", 300, 2)
    assert a.result_name == "eval_opd_chartqa_human_q300_s1.json" and a.result_name != b.result_name
    assert a.merged_repo.endswith("_q300_s1_merged") and a.ckpt_repo.endswith("_q300_s1_ckpt")
    assert a.ood_result_name("ffyang/vlm_opd_charxiv") == "eval_opd_chartqa_human_q300_s1_on_vlm_opd_charxiv.json"
    with pytest.raises(ValueError):
        point_names("chartqa_human", t, d, "gkd", 300, 1)


def test_commands_carry_the_config():
    cfg = load_tasks(TASKS_FILE)
    d, t = cfg["defaults"], cfg["tasks"]["chartqa_human"]
    n = point_names("chartqa_human", t, d, "opd", 300, 1)
    cmd = train_command("opd", n, t, d, 300, 1, d["student"], d["teacher"], None, None)
    assert "vlm_opd.opd_trainer" in cmd and "--limit" in cmd and cmd[cmd.index("--limit") + 1] == "300"
    assert cmd[cmd.index("--batch-size") + 1] == "16" and cmd[cmd.index("--total-steps") + 1] == "150"
    cmd = train_command("sft", point_names("chartqa_human", t, d, "sft", 300, 1), t, d, 300, 1, d["student"], d["teacher"], 20, None)
    assert "vlm_opd.sft" in cmd and cmd[cmd.index("--max-steps") + 1] == "20"
    assert cmd[cmd.index("--max-question-index") + 1] == "300"
    e = eval_command("repo/m", "repo/d", "outputs/x.json", "tag", 1, 0.85)
    assert "vlm_opd.evaluate" in e and "--split" in e


def test_local_artifact_mode_keeps_merged_models_off_the_hub():
    from vlm_opd.experiment import eval_model, merged_mode

    cfg = load_tasks(TASKS_FILE)
    d, t = cfg["defaults"], cfg["tasks"]["chartqa_human"]
    assert merged_mode(d) == "local"
    n = point_names("chartqa_human", t, d, "opd", 300, 1)
    cmd = train_command("opd", n, t, d, 300, 1, d["student"], d["teacher"], None, None)
    assert "--merged-repo" not in cmd and cmd[cmd.index("--ckpt-repo") + 1] == n.ckpt_repo
    s = point_names("chartqa_human", t, d, "sft", 300, 1)
    cmd = train_command("sft", s, t, d, 300, 1, d["student"], d["teacher"], None, None)
    assert "--push-merged-repo" not in cmd and "--merge" in cmd
    assert cmd[cmd.index("--push-adapter-repo") + 1] == "ffyang/vlm_opd_sft_chartqa_human_q300_s1_lora"
    assert eval_model(n, d) == "ckpt/opd_chartqa_human_q300_s1/merged"
    hub = {**d, "artifacts": {"merged": "hub"}}
    assert eval_model(n, hub) == n.merged_repo
    assert "--merged-repo" in train_command("opd", n, t, hub, 300, 1, d["student"], d["teacher"], None, None)
    with pytest.raises(ValueError):
        merged_mode({**d, "artifacts": {"merged": "s3"}})


def test_dry_run_builds_train_then_eval(tmp_path):
    plan = run_point("chartqa_human", "opd", 100, 7, dry_run=True, opd_steps=5, skip_ood=True)
    modules = [c[2] for c in plan["commands"]]
    assert modules == ["vlm_opd.opd_trainer", "vlm_opd.evaluate"]
    assert plan["commands"][0][plan["commands"][0].index("--prompt-style") + 1] == "chart"
    assert plan["commands"][1][plan["commands"][1].index("--model") + 1] == "ckpt/opd_chartqa_human_q100_s7/merged"
    with_ood = run_point("chartqa_human", "opd", 100, 7, dry_run=True, opd_steps=5)
    n_ood = len(load_tasks(TASKS_FILE)["tasks"]["chartqa_human"]["ood_eval"])
    assert [c[2] for c in with_ood["commands"]] == ["vlm_opd.opd_trainer"] + ["vlm_opd.evaluate"] * (1 + n_ood)
    assert "5" == plan["commands"][0][plan["commands"][0].index("--total-steps") + 1]
