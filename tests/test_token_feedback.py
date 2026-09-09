"""score_batch on a tiny Qwen3-VL: shapes, masks, and per-token records line up."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_pair():
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from vlm_opd.common import STUDENT_MODEL, image_pixel_bounds
    from vlm_opd.modeling import freeze_all

    try:
        processor = AutoProcessor.from_pretrained(STUDENT_MODEL, **image_pixel_bounds())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"processor unavailable: {type(e).__name__}")
    tok = processor.tokenizer
    text_cfg = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "vocab_size": len(tok), "max_position_embeddings": 4096, "head_dim": 16, "rope_theta": 10000.0}
    vision_cfg = {"depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "out_hidden_size": 64,
                  "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3, "deepstack_visual_indexes": [0, 1]}
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg, image_token_id=processor.image_token_id,
                        video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
                        vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
                        vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"))
    torch.manual_seed(0); teacher = Qwen3VLForConditionalGeneration(cfg).eval(); freeze_all(teacher)
    torch.manual_seed(1); student = Qwen3VLForConditionalGeneration(cfg).eval(); freeze_all(student)
    rows = [{"id": "t0", "image": Image.new("RGB", (768, 512)), "question": "Value in 2020?", "answer": "45"},
            {"id": "t1", "image": Image.new("RGB", (512, 640)), "question": "Highest country?", "answer": "Germany"},
            {"id": "t2", "image": Image.new("RGB", (640, 640)), "question": "Sum of bars?", "answer": "12"}]
    return processor, teacher, student, rows


def test_score_batch_records(tiny_pair):
    from vlm_opd.analysis.token_classes import CLASSES
    from vlm_opd.analysis.token_feedback import score_batch

    processor, teacher, student, rows = tiny_pair
    recs = score_batch(student, teacher, processor, rows, max_new_tokens=6, temperature=1.0, seed=3, micro=2)
    assert [r["id"] for r in recs] == ["t0", "t1", "t2"]
    for r in recs:
        n = r["n_tokens"]
        assert 0 < n <= 6
        assert len(r["pieces"]) == len(r["kl"]) == len(r["logratio"]) == len(r["classes"]) == n
        assert all(v >= 0 for v in r["kl"])
        assert all(c in CLASSES for c in r["classes"])
        # log-ratio equals the difference of the recorded log-probabilities
        for a, ls, lt in zip(r["logratio"], r["logp_student"], r["logp_teacher"]):
            assert abs(a - (ls - lt)) < 1e-3
        assert isinstance(r["correct"], bool)
