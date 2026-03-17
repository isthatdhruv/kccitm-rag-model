"""LLM Client smoke test — requires Ollama running with qwen3:8b and nomic-embed-text.

Usage:
    python -m tests.test_llm_client              # Run all tests
    python -m tests.test_llm_client 3            # Run only test 3
    python -m tests.test_llm_client 3 5 7        # Run tests 3, 5, and 7
    python -m tests.test_llm_client --from 4     # Run tests 4 through 7
"""

import argparse
import asyncio
import json
import sys

from core.llm_client import OllamaClient


async def test_health(llm: OllamaClient) -> None:
    """Test 1: Health check."""
    health = await llm.health_check()
    assert health["status"] == "ok", f"Ollama not running: {health}"
    print(f"  \u2713 Ollama running. Models: {health['models']}")


async def test_generate(llm: OllamaClient) -> None:
    """Test 2: Simple generation."""
    response = await llm.generate("What is 2+2? Reply with just the number.")
    assert "4" in response, f"Unexpected response: {response}"
    print(f"  \u2713 Generation works: '{response.strip()[:50]}'")


async def test_json_format(llm: OllamaClient) -> None:
    """Test 3: JSON format generation."""
    response = await llm.generate(
        'Return a JSON object with keys "name" and "age" for a 25-year-old named Alice.',
        format="json",
        temperature=0.1,
    )
    parsed = json.loads(response)
    assert "name" in parsed and "age" in parsed, f"Bad JSON: {response}"
    print(f"  \u2713 JSON mode works: {parsed}")


async def test_chat(llm: OllamaClient) -> None:
    """Test 4: Chat completion."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be very brief."},
        {"role": "user", "content": "What is Python?"},
    ]
    response = await llm.chat(messages)
    assert len(response) > 10, f"Response too short: {response}"
    print(f"  \u2713 Chat works: '{response.strip()[:80]}...'")


async def test_streaming(llm: OllamaClient) -> None:
    """Test 5: Streaming."""
    messages = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]
    tokens: list[str] = []
    async for token in llm.stream_chat(messages):
        tokens.append(token)
    full = "".join(tokens)
    assert "1" in full and "5" in full, f"Streaming incomplete: {full}"
    print(f"  \u2713 Streaming works: {len(tokens)} tokens received")


async def test_embedding(llm: OllamaClient) -> None:
    """Test 6: Embedding."""
    embedding = await llm.embed("Hello world")
    assert len(embedding) == 768, f"Expected 768-dim, got {len(embedding)}"
    assert all(isinstance(x, float) for x in embedding[:5])
    print(f"  \u2713 Embedding works: {len(embedding)}-dim vector")


async def test_batch_embedding(llm: OllamaClient) -> None:
    """Test 7: Batch embedding."""
    embeddings = await llm.embed_batch(["Hello", "World", "Test"])
    assert len(embeddings) == 3
    assert all(len(e) == 768 for e in embeddings)
    print(f"  \u2713 Batch embedding works: {len(embeddings)} vectors")


# Ordered registry — index+1 = test number shown to user
TESTS = [
    test_health,
    test_generate,
    test_json_format,
    test_chat,
    test_streaming,
    test_embedding,
    test_batch_embedding,
]


async def run_tests(test_nums: list[int]) -> None:
    llm = OllamaClient()
    passed = 0
    failed = 0

    for num in test_nums:
        fn = TESTS[num - 1]
        label = fn.__doc__ or fn.__name__
        print(f"{label}")
        try:
            await fn(llm)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  \033[91m\u2717 FAILED: {e}\033[0m")
        print()

    if failed:
        print(f"\033[91m\u2717 {passed}/{passed + failed} tests passed\033[0m")
        sys.exit(1)
    else:
        print(f"\033[92m\u2713 All {passed} tests passed!\033[0m")


def parse_args() -> list[int]:
    parser = argparse.ArgumentParser(description="LLM Client smoke tests")
    parser.add_argument("tests", nargs="*", type=int, help="Test numbers to run (e.g. 3 5 7)")
    parser.add_argument("--from", dest="start", type=int, help="Run from this test number onward")
    args = parser.parse_args()

    all_nums = list(range(1, len(TESTS) + 1))

    if args.start:
        if args.start < 1 or args.start > len(TESTS):
            parser.error(f"--from must be between 1 and {len(TESTS)}")
        return [n for n in all_nums if n >= args.start]

    if args.tests:
        for t in args.tests:
            if t < 1 or t > len(TESTS):
                parser.error(f"Test {t} does not exist (valid: 1-{len(TESTS)})")
        return args.tests

    return all_nums


if __name__ == "__main__":
    asyncio.run(run_tests(parse_args()))
