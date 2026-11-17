"""Grounded answer generation with Anthropic Claude.

Production concerns handled here:

* **Grounding** — the system prompt restricts the model to the numbered sources and to an exact
  refusal string; citations in the answer are parsed and checked against the sources we sent.
* **Context budget** — retrieved chunks are added in rank order until ``max_context_chars``.
* **Cost / latency** — token usage from the API response, wall-clock latency, and an estimated
  USD cost from a price table (overridable by env vars) are returned with every answer.
* **Failure modes** — explicit timeout; retries with exponential backoff + jitter on rate limits,
  connection errors and 5xx/overloaded; no retry on other 4xx; a typed error the API maps to 503.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from antitrust_rag.models import ScoredChunk

logger = logging.getLogger(__name__)

NOT_FOUND = "Not found in sources."

SYSTEM_PROMPT = f"""You are a careful legal research assistant for competition (antitrust) law.

Rules:
1. Answer ONLY with information stated in the <sources> provided by the user. Do not use outside
   knowledge, case law or later amendments that are not in the sources.
2. Support every sentence with at least one citation in square brackets, copied from the
   `citation` attribute of the source you used, e.g. [Lei 12.529/2011, Art. 36, §3] or
   [TFEU, Art. 102]. You may narrow a citation to a specific item (e.g. "§3, I") when the
   source text shows it. Never cite a source you were not given.
3. If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
   If they answer only part of the question, answer that part and say what is missing.
4. Answer in the language of the question. Quote short key phrases from the law when useful
   (the sources may be in Portuguese or English).
5. Be concise: at most ~200 words. This is legal information, not legal advice."""

# USD per million tokens (input, output), from Anthropic's public pricing page as of Nov 2026.
# Prices change: check https://www.anthropic.com/pricing and override with
# RAG_LLM_PRICE_INPUT_PER_MTOK / RAG_LLM_PRICE_OUTPUT_PER_MTOK. Prompt-cache and batch discounts
# are not modelled. Unknown models get cost=None rather than a guessed number.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_CITATION_RE = re.compile(r"\[([^\[\]]+?,\s*Art\.[^\[\]]*)\]")
_ARTICLE_RE = re.compile(r"Art\.\s*(\d+(?:-[A-Z])?)")


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    attempts: int = 1


class LLMError(RuntimeError):
    """Raised when the LLM call failed after retries (or is not configured)."""


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse: ...


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    override: tuple[float | None, float | None] = (None, None),
) -> float | None:
    price_in, price_out = override
    if price_in is None or price_out is None:
        # longest matching prefix, so dated ids ("...-20251001") resolve to their family
        names = sorted((n for n in PRICES_PER_MTOK if model.startswith(n)), key=len, reverse=True)
        base = PRICES_PER_MTOK[names[0]] if names else None
        if base is None:
            return None  # unknown model: report tokens, but do not invent a price
        price_in = base[0] if price_in is None else price_in
        price_out = base[1] if price_out is None else price_out
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


class ClaudeClient:
    """Thin wrapper over the official ``anthropic`` SDK with explicit retry/timeout policy."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        if client is None:
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            # SDK-level retries disabled: we retry ourselves so attempts are logged and bounded.
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._client = client

    def _retryable(self, exc: Exception) -> bool:
        a = self._anthropic
        return isinstance(exc, a.RateLimitError | a.APIConnectionError | a.InternalServerError)

    def _backoff(self, attempt: int, exc: Exception) -> float:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after", ""))
            except (TypeError, ValueError):
                retry_after = None
        delay = retry_after if retry_after is not None else 0.5 * 2**attempt
        return min(delay, 20.0) + random.uniform(0, 0.25)

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        t0 = time.perf_counter()
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
                ).strip()
                return LLMResponse(
                    text=text,
                    model=getattr(msg, "model", self.model),
                    input_tokens=int(msg.usage.input_tokens),
                    output_tokens=int(msg.usage.output_tokens),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    attempts=attempt + 1,
                )
            except Exception as exc:
                last = exc
                if not self._retryable(exc) or attempt == self.max_retries:
                    break
                delay = self._backoff(attempt, exc)
                logger.warning(
                    "llm call failed, retrying",
                    extra={"attempt": attempt + 1, "delay_s": round(delay, 2), "error": repr(exc)},
                )
                self._sleep(delay)
        raise LLMError(f"LLM call failed: {last!r}") from last


# ---------------------------------------------------------------------------------------------
# Prompt assembly and answer post-processing
# ---------------------------------------------------------------------------------------------


def build_context(chunks: list[ScoredChunk], max_chars: int) -> tuple[str, list[ScoredChunk]]:
    """Render sources as XML-ish blocks, in rank order, until the character budget is spent."""
    used: list[ScoredChunk] = []
    blocks: list[str] = []
    total = 0
    for sc in chunks:
        block = (
            f'<source id="{len(used) + 1}" citation="{sc.chunk.citation}">\n'
            f"{sc.chunk.text}\n</source>"
        )
        if used and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(sc)
        total += len(block)
    return "<sources>\n" + "\n".join(blocks) + "\n</sources>", used


def build_user_prompt(question: str, context: str) -> str:
    return f"{context}\n\nQuestion: {question}"


@dataclass
class CitationCheck:
    text: str
    supported: bool


@dataclass
class Answer:
    answer: str
    grounded: bool
    citations: list[CitationCheck] = field(default_factory=list)
    sources: list[ScoredChunk] = field(default_factory=list)
    usage: LLMResponse | None = None
    cost_usd: float | None = None


def check_citations(answer: str, sources: list[ScoredChunk]) -> list[CitationCheck]:
    """A citation is *supported* if its source title and article match a chunk we sent."""
    allowed = {(s.chunk.source_title, s.chunk.unit.article) for s in sources}
    seen: dict[str, CitationCheck] = {}
    for raw in _CITATION_RE.findall(answer):
        for part in raw.split(";"):  # "[A, Art. 1; B, Art. 2]"
            part = part.strip()
            title = part.split(",", 1)[0].strip()
            art = _ARTICLE_RE.search(part)
            ok = bool(art) and (title, art.group(1) if art else "") in allowed
            seen.setdefault(part, CitationCheck(part, ok))
    return list(seen.values())


def is_not_found(text: str) -> bool:
    return text.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower()
