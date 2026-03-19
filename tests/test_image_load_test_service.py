from src.services.image_load_test_service import ImageLoadTestService


def test_build_prompt_does_not_append_loadtest_sequence_marker():
    prompt = ImageLoadTestService._build_prompt(index=2, prompt_prefix="")

    assert "第 3 次并发自测样本" not in prompt
    assert "雨夜霓虹街头的拉面小店门口" in prompt


def test_build_prompt_keeps_prefix_without_injecting_sequence_marker():
    prompt = ImageLoadTestService._build_prompt(index=9, prompt_prefix="高清壁纸风格")

    assert prompt.startswith("高清壁纸风格，")
    assert "热带雨林中的巨大白色瀑布" in prompt
    assert "并发自测样本" not in prompt
