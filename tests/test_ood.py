import json

from vlm_opd.analysis.ood import collect_ood, find_ood_runs, markdown_ood


def _write(path, correct):
    path.write_text(json.dumps({"records": [{"id": f"t{i:03d}", "correct": c} for i, c in enumerate(correct)]}))


def test_ood_table_pairs_by_seed_and_lists_zeroshot(tmp_path):
    n = 40
    base = [i % 3 == 0 for i in range(n)]
    _write(tmp_path / "eval_student_zeroshot_on_vlm_opd_ood_charxiv.json", base)
    _write(tmp_path / "eval_teacher_zeroshot_on_vlm_opd_ood_charxiv.json", [i % 3 != 2 for i in range(n)])
    for s in (1, 2):
        _write(tmp_path / f"eval_sft_chartqa_human_q300_s{s}_on_vlm_opd_ood_charxiv.json", [i % 2 == 0 for i in range(n)])
        _write(tmp_path / f"eval_opd_chartqa_human_q300_s{s}_on_vlm_opd_ood_charxiv.json", [i % 4 != 3 for i in range(n)])
        _write(tmp_path / f"eval_sft_chartqa_human_q300_s{s}.json", [i % 2 == 0 for i in range(n)])
        _write(tmp_path / f"eval_opd_chartqa_human_q300_s{s}.json", [i % 4 != 3 for i in range(n)])
    _write(tmp_path / "eval_opd_chartqa_human_q300_s99_on_vlm_opd_ood_charxiv.json", [True] * n)  # smoke seed, ignored
    runs = find_ood_runs(tmp_path, "chartqa_human", "vlm_opd_ood_charxiv")
    assert sorted(runs[("opd", 300)]) == [1, 2]
    table = collect_ood(tmp_path, "chartqa_human", [300], n_boot=200)
    block = next(b for b in table["blocks"] if b["set"] == "vlm_opd_ood_charxiv")
    assert set(block["zeroshot"]) == {"student", "teacher"}
    row = block["points"][0]
    assert row["opd"]["accuracy"] == 0.75 and row["sft"]["accuracy"] == 0.5 and row["opd_id"] == 0.75
    assert row["opd_minus_sft"]["seeds"] == [1, 2] and abs(row["opd_minus_sft"]["delta"] - 0.25) < 1e-9
    md = markdown_ood(table)
    assert "### vlm_opd_ood_charxiv" in md and "student zero-shot" in md and "ID 0.750" in md
    other = next(b for b in table["blocks"] if b["set"] == "vlm_opd_ood_chartqapro")
    assert other["points"][0]["opd"] is None
