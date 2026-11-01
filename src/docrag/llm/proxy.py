"""LLMProxy: the single gateway every LLM call goes through.

route "provider:model" -> Provider, plus, written ONCE for all providers:
* cache   (re-runs are free and reproducible),
* cost    (sum of usage * price) and a hard budget cap,
* structured output: strip think tags -> Pydantic validation -> exactly ONE
  retry with the validation error -> otherwise a hard, counted failure.

Why our own proxy instead of LiteLLM? LiteLLM does routing + format conversion for
100+ providers and is the pragmatic choice in a product. Here the gateway logic
(format mapping, message order, cost accounting, caching, later tracing) is the
thing being demonstrated, and with 2-3 providers it is ~300 lines we fully own
and test. If the provider list grew, swapping the adapters for LiteLLM would be
a local change behind the same Provider interface.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from docrag.llm.base import Provider
from docrag.llm.pricing import cost_eur
from docrag.llm.types import ChatCompletion, Message, json_schema_for
from docrag.shared import timing, tracing
from docrag.shared.cache import Cache, stable_hash
from docrag.shared.settings import Settings
from docrag.shared.timing import timed

log = logging.getLogger(__name__)


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class BudgetExceededError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


@dataclass
class ProxyStats:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # cost_eur = what this process was BILLED (cache hits are free); the budget
    # cap uses it. cost_eur_full = what the same calls cost on a cold cache
    # (cache hits priced from their stored usage), the number to compare runs
    # by: a re-run with 90% cache hits otherwise looks 10x cheaper.
    cost_eur: float = 0.0
    cost_eur_full: float = 0.0
    validation_retries: int = 0
    validation_failures: int = 0
    per_route_calls: dict[str, int] = field(default_factory=dict)


class LLMProxy:
    def __init__(
        self,
        providers: dict[str, Provider],
        cache: Cache | None = None,
        budget_eur: float = 5.0,
    ) -> None:
        self.providers = providers
        self.cache = cache
        self.budget_eur = budget_eur
        self.stats = ProxyStats()

    def resolve(self, route: str) -> tuple[Provider, str]:
        # partition on the FIRST ":" only: "ollama:qwen3:8b" -> ("ollama", "qwen3:8b")
        provider_name, sep, model = route.partition(":")
        if not sep or not model:
            raise ValueError(f"route must look like 'provider:model', got {route!r}")
        if provider_name not in self.providers:
            raise ValueError(
                f"provider {provider_name!r} not configured (missing API key?); "
                f"available: {sorted(self.providers)}"
            )
        return self.providers[provider_name], model

    @timed("llm", kind=tracing.LLM)
    async def chat(
        self,
        route: str,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
        temperature: float | None = 0.0,
    ) -> ChatCompletion:
        _check_order(messages)
        provider, model = self.resolve(route)
        _trace_request(route, model, messages, response_format, temperature)
        schema = json_schema_for(response_format) if response_format else None
        # Content-addressed key: any change in prompt, context, model or schema
        # yields a new key. No secrets in it.
        key = stable_hash(
            route, [m.model_dump() for m in messages], schema, temperature
        )
        if self.cache is not None:
            hit = self.cache.get_json("llm", key)
            if hit is not None:
                self.stats.cache_hits += 1
                timing.cache_hit()
                cached = ChatCompletion.model_validate(hit)
                self.stats.cost_eur_full += cost_eur(route, cached.usage)
                _trace_response(route, cached, cache_hit=True)
                return cached

        if self.stats.cost_eur >= self.budget_eur:
            raise BudgetExceededError(
                f"spent {self.stats.cost_eur:.2f} EUR, "
                f"budget is {self.budget_eur:.2f} EUR"
            )
        completion = await provider.chat(model, messages, response_format, temperature)
        self._account(route, completion)
        _trace_response(route, completion, cache_hit=False)
        if self.cache is not None:
            self.cache.set_json("llm", key, completion.model_dump())
        return completion

    async def structured[T: BaseModel](
        self,
        route: str,
        messages: list[Message],
        schema: type[T],
        temperature: float | None = 0.0,
    ) -> tuple[T, ChatCompletion]:
        """Call the LLM and parse the reply into ``schema``.

        Invalid JSON gets exactly one retry that includes the validation error.
        """
        completion = await self.chat(route, messages, schema, temperature)
        try:
            return _parse(completion.content, schema), completion
        except ValidationError as err:
            self.stats.validation_retries += 1
            tracing.annotate({"docrag.validation_retry": True})
            log.warning(
                "invalid %s from %s, retrying once: %s", schema.__name__, route, err
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=completion.content),
                Message(
                    role="user",
                    content=(
                        "Your reply did not match the required JSON schema:\n"
                        f"{err}\nReply again with ONLY valid JSON for the schema."
                    ),
                ),
            ]
        completion = await self.chat(route, retry_messages, schema, temperature)
        try:
            return _parse(completion.content, schema), completion
        except ValidationError as err:
            self.stats.validation_failures += 1
            raise StructuredOutputError(
                f"{route}: invalid {schema.__name__} after retry"
            ) from err

    def _account(self, route: str, completion: ChatCompletion) -> None:
        s = self.stats
        s.calls += 1
        s.per_route_calls[route] = s.per_route_calls.get(route, 0) + 1
        s.input_tokens += completion.usage.input_tokens
        s.output_tokens += completion.usage.output_tokens
        cost = cost_eur(route, completion.usage)
        s.cost_eur += cost
        s.cost_eur_full += cost

    def stats_dict(self) -> dict[str, float | int]:
        """The numbers that go into run.json."""
        s = self.stats
        return {
            "calls": s.calls,
            "cache_hits": s.cache_hits,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "cost_eur_billed": round(s.cost_eur, 4),
            "cost_eur_full": round(s.cost_eur_full, 4),
            "validation_retries": s.validation_retries,
            "validation_failures": s.validation_failures,
        }

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()

    @classmethod
    def from_settings(
        cls, settings: Settings, cache: Cache | None = None
    ) -> "LLMProxy":
        from docrag.llm.anthropic_provider import AnthropicProvider
        from docrag.llm.fake import FakeProvider
        from docrag.llm.openai_compat import OpenAICompatProvider

        providers: dict[str, Provider] = {
            "fake": FakeProvider(),
            "ollama": OpenAICompatProvider("ollama", settings.ollama_base_url),
        }
        # Only providers with a key are registered; a route to a missing one fails
        # with a clear message in resolve().
        openai_compatible = {
            "openai": ("https://api.openai.com/v1", settings.openai_api_key),
            "groq": ("https://api.groq.com/openai/v1", settings.groq_api_key),
            "gemini": (
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                settings.gemini_api_key,
            ),
            "mistral": ("https://api.mistral.ai/v1", settings.mistral_api_key),
            "deepseek": ("https://api.deepseek.com/v1", settings.deepseek_api_key),
        }
        for name, (url, key) in openai_compatible.items():
            if key is not None:
                providers[name] = OpenAICompatProvider(
                    name, url, key.get_secret_value()
                )
        # OpenRouter: one key, hundreds of models, OpenAI format.
        # Route: "openrouter:anthropic/claude-haiku-5-5" (model = vendor/model).
        if settings.openrouter_api_key is not None:
            providers["openrouter"] = OpenAICompatProvider(
                "openrouter",
                "https://openrouter.ai/api/v1",
                settings.openrouter_api_key.get_secret_value(),
                extra_body={
                    # Only route to upstream hosts that support every parameter
                    # we send - above all response_format/json_schema. Otherwise
                    # a request may land on a host that ignores the schema.
                    "provider": {"require_parameters": True},
                    # Ask OpenRouter to report the billed cost in `usage`.
                    "usage": {"include": True},
                },
            )
        if settings.anthropic_api_key is not None:
            providers["anthropic"] = AnthropicProvider(
                settings.anthropic_api_key.get_secret_value()
            )
        return cls(providers, cache=cache, budget_eur=settings.budget_eur)


def _trace_request(
    route: str,
    model: str,
    messages: list[Message],
    response_format: type[BaseModel] | None,
    temperature: float | None,
) -> None:
    """OpenInference LLM attributes for the request (no-op outside a trace)."""
    attrs: dict[str, object] = {
        "llm.provider": route.partition(":")[0],
        "llm.model_name": model,
        "llm.invocation_parameters": json.dumps(
            {
                "temperature": temperature,
                "response_format": response_format.__name__
                if response_format
                else None,
            }
        ),
    }
    for i, m in enumerate(messages):
        attrs[f"llm.input_messages.{i}.message.role"] = m.role
        # .text drops images: base64 page renders don't belong in a span.
        attrs[f"llm.input_messages.{i}.message.content"] = m.text
    tracing.annotate(attrs)


def _trace_response(route: str, completion: ChatCompletion, cache_hit: bool) -> None:
    u = completion.usage
    tracing.annotate(
        {
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": completion.content,
            "output.value": completion.content,
            "llm.token_count.prompt": u.input_tokens,
            "llm.token_count.completion": u.output_tokens,
            "llm.token_count.total": u.input_tokens + u.output_tokens,
            # A cache hit costs nothing now, but we keep the tokens: they show
            # what the call WOULD cost on a cold cache.
            "docrag.cache_hit": cache_hit,
            "docrag.cost_eur": 0.0 if cache_hit else cost_eur(route, u),
        }
    )


def _parse[T: BaseModel](content: str, schema: type[T]) -> T:
    # The only normalization we allow: unwrap a ```json fence. No regex "JSON
    # repair" - if it's still invalid, the retry with the error message decides.
    text = content.strip()
    fence = _FENCE.match(text)
    if fence:
        text = fence.group(1)
    return schema.model_validate_json(text)


def _check_order(messages: list[Message]) -> None:
    """System messages first, and at least one non-system message.

    Providers disagree on what they accept (Anthropic has a separate `system`
    field, some local chat templates break on a system message in the middle),
    so we enforce one portable order here instead of debugging it per provider.
    """
    seen_other = False
    for m in messages:
        if m.role == "system" and seen_other:
            raise ValueError("system messages must come before user/assistant messages")
        seen_other = seen_other or m.role != "system"
    if not seen_other:
        raise ValueError("need at least one user message")
