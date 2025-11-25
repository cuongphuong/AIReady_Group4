#!/usr/bin/env python
"""
Quick smoke test for async batch classifier
"""
import asyncio
import json
from bug_classifier import batch_classify, classify_bug

async def main():
    print("=" * 60)
    print("Testing async batch_classify implementation")
    print("=" * 60)

    # Test 1: Single item via heuristic (quick path)
    print("\n[Test 1] Single bug - heuristic match (UI keyword)")
    bug1 = "Nút bị lệch vị trí trên giao diện"
    result1 = await classify_bug(bug1)
    print(f"Input: {bug1}")
    print(f"Result: {json.dumps(result1, ensure_ascii=False, indent=2)}")
    assert result1.get('label') == 'UI', f"Expected UI, got {result1.get('label')}"
    print("✓ PASS")

    # Test 2: Batch with mixed heuristic and LLM (if API is available, will use heuristic)
    print("\n[Test 2] Batch classify - 3 bugs")
    bugs = [
        "Nút 'Save' không click được, lỗi JavaScript trên console",
        "Format ngày hiển thị sai MM/DD thay vì DD/MM",
        "Service timeout khi gọi API bên thứ 3"
    ]
    results = await batch_classify(bugs)
    print(f"Input: {len(bugs)} bugs")
    for i, (bug, res) in enumerate(zip(bugs, results)):
        print(f"\n  [{i}] {bug[:50]}...")
        print(f"      Label: {res.get('label')}, Team: {res.get('team')}")
        assert res is not None, f"Result {i} is None"
        assert 'label' in res, f"Result {i} missing 'label'"
    print("✓ PASS - All batch items classified")

    # Test 3: Empty list edge case
    print("\n[Test 3] Empty batch")
    empty_results = await batch_classify([])
    print(f"Empty batch result: {empty_results}")
    assert len(empty_results) == 0, "Expected empty result for empty input"
    print("✓ PASS")

    # Test 4: Verify retry mechanism is in place (no exception on transient errors would be harder to test without mocking)
    print("\n[Test 4] Verify async helpers are callable")
    from bug_classifier import _quick_heuristic_for_text, _call_model_with_retries
    h = _quick_heuristic_for_text("backend 500 error")
    print(f"Heuristic result for 'backend 500 error': {h}")
    assert h is not None and h.get('label') == 'Backend', "Heuristic should match Backend"
    print("✓ PASS")

    print("\n" + "=" * 60)
    print("All smoke tests PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
