from speculators.data_generation.preprocessing import (
    _adapt_part_for_vllm,
    _preprocess_batch,
)


def test_vllm_local_media_url_uses_encoded_file_uri(tmp_path):
    media_path = tmp_path / "中文 image.png"

    part = _adapt_part_for_vllm({"type": "image", "path": str(media_path)})

    assert part == {
        "type": "image_url",
        "image_url": {"url": media_path.absolute().as_uri()},
    }
    assert "%20" in part["image_url"]["url"]
    assert "%E4%B8%AD%E6%96%87" in part["image_url"]["url"]


def test_pretokenized_rows_are_truncated_instead_of_dropped():
    results = _preprocess_batch(
        {
            "input_ids": [[1, 2, 3], [1, 2, 3, 4, 5]],
            "loss_mask": [[0, 1, 1], [0, 0, 1, 1, 1]],
        },
        processor=None,  # type: ignore[arg-type]
        max_length=4,
        assistant_pattern=None,
    )

    assert [row.tolist() for row in results["input_ids"]] == [
        [1, 2, 3],
        [1, 2, 3, 4],
    ]
