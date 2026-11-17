from types import SimpleNamespace

import anthropic
import httpx
import pytest

from antitrust_rag.generation import (
    NOT_FOUND,
    SYSTEM_PROMPT,
    ClaudeClient,
    LLMError,
    build_context,
    check_citations,
    estimate_cost_usd,
    is_not_found,
)
from antitrust_rag.models import Chunk, LegalUnit, ScoredChunk

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _message(text: str = "Answer [TFEU, Art. 102].") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        model="claude-sonnet-5-5",
        usage=SimpleNamespace(input_tokens=1200, output_tokens=150),
    )


class FakeMessages:
    """Mimics ``anthropic.Anthropic().messages`` — raises queued errors, then succeeds."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return _message()


def _client(errors=None, retries=3):
    fake = SimpleNamespace(messages=FakeMessages(errors))
    sleeps: list[float] = []
    c = ClaudeClient(
        None, "claude-sonnet-5-5", client=fake, max_retries=retries, sleep=sleeps.append
    )
    return c, fake.messages, sleeps


def _rate_limit() -> anthropic.RateLimitError:
    resp = httpx.Response(429, request=REQ, headers={"retry-after": "1"})
    return anthropic.RateLimitError("slow down", response=resp, body=None)


def _bad_request() -> anthropic.BadRequestError:
    return anthropic.BadRequestError("bad", response=httpx.Response(400, request=REQ), body=None)


def test_complete_success_reports_usage() -> None:
    c, msgs, sleeps = _client()
    r = c.complete(SYSTEM_PROMPT, "question", 300)
    assert r.text == "Answer [TFEU, Art. 102]."
    assert (r.input_tokens, r.output_tokens, r.attempts) == (1200, 150, 1)
    call = msgs.calls[0]
    assert call["system"] == SYSTEM_PROMPT and call["max_tokens"] == 300
    assert sleeps == []


def test_retries_transient_errors_with_backoff() -> None:
    c, _msgs, sleeps = _client([_rate_limit(), anthropic.APIConnectionError(request=REQ)])
    r = c.complete("s", "u", 10)
    assert r.attempts == 3
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 1.3  # honours Retry-After (+ jitter)


def test_does_not_retry_client_errors() -> None:
    c, msgs, sleeps = _client([_bad_request()])
    with pytest.raises(LLMError):
        c.complete("s", "u", 10)
    assert len(msgs.calls) == 1 and sleeps == []


def test_gives_up_after_max_retries() -> None:
    c, msgs, _ = _client([anthropic.APITimeoutError(request=REQ)] * 5, retries=2)
    with pytest.raises(LLMError):
        c.complete("s", "u", 10)
    assert len(msgs.calls) == 3


def test_missing_api_key_is_a_clear_error() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(None, "claude-sonnet-5-5")


def test_cost_estimation() -> None:
    assert estimate_cost_usd("claude-sonnet-5-5", 1_000_000, 0) == 2.0
    assert estimate_cost_usd("claude-sonnet-4-6", 0, 1_000_000) == 15.0
    assert estimate_cost_usd("claude-haiku-4-5-20251001", 1000, 1000) == pytest.approx(0.006)
    assert estimate_cost_usd("some-future-model", 10, 10) is None
    assert estimate_cost_usd("some-future-model", 1_000_000, 0, (2.0, 8.0)) == 2.0


def _sc(title: str, art: str, para: str | None = None, text: str = "t" * 50) -> ScoredChunk:
    return ScoredChunk(Chunk("s", title, LegalUnit(art, para), text), 1.0)


def test_build_context_respects_budget_and_numbers_sources() -> None:
    chunks = [_sc("Lei 12.529/2011", str(i), text="x" * 400) for i in range(10)]
    ctx, used = build_context(chunks, max_chars=1000)
    assert 1 <= len(used) < 10
    assert '<source id="1" citation="Lei 12.529/2011, Art. 0">' in ctx
    # the first source is always included, even if it alone exceeds the budget
    _, used_one = build_context([_sc("A", "1", text="y" * 5000)], max_chars=100)
    assert len(used_one) == 1


def test_check_citations_flags_unsupported_ones() -> None:
    used = [_sc("Lei 12.529/2011", "36", "3"), _sc("TFEU", "102")]
    answer = (
        "Bid rigging is an infringement [Lei 12.529/2011, Art. 36, §3, I, d)]. "
        "Abuse is prohibited [TFEU, Art. 102(a)]. Invented [Lei 12.529/2011, Art. 999]. "
        "Two at once [TFEU, Art. 102; Lei 12.529/2011, Art. 36]."
    )
    checks = {c.text: c.supported for c in check_citations(answer, used)}
    assert checks["Lei 12.529/2011, Art. 36, §3, I, d)"] is True
    assert checks["TFEU, Art. 102(a)"] is True
    assert checks["Lei 12.529/2011, Art. 999"] is False
    assert checks["Lei 12.529/2011, Art. 36"] is True


def test_not_found_detection() -> None:
    assert is_not_found(NOT_FOUND)
    assert is_not_found("  not found in sources ")
    assert not is_not_found("The answer is not found in sources such as X, but ...")


def test_real_sdk_wire_format_with_mocked_http() -> None:
    """Drive the *real* anthropic SDK against a mocked HTTP transport (no network, no key)."""
    import json

    try:  # anthropic>=2 ships its own fork of httpx; older SDKs use httpx itself
        import httpx2 as sdk_httpx  # type: ignore[import-not-found]
    except ImportError:
        sdk_httpx = httpx

    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return sdk_httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "Prohibited [TFEU, Art. 102]."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 321, "output_tokens": 12},
            },
        )

    sdk = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=sdk_httpx.Client(transport=sdk_httpx.MockTransport(handler)),
    )
    r = ClaudeClient(None, "claude-sonnet-5-5", client=sdk).complete(SYSTEM_PROMPT, "Q?", 200)
    assert captured["url"].endswith("/v1/messages")
    assert captured["body"]["model"] == "claude-sonnet-5-5"
    assert captured["body"]["system"] == SYSTEM_PROMPT
    assert captured["body"]["messages"] == [{"role": "user", "content": "Q?"}]
    assert (r.text, r.input_tokens, r.output_tokens) == ("Prohibited [TFEU, Art. 102].", 321, 12)
