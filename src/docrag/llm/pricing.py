"""Price table in USD per 1M tokens (input, output), and the EUR conversion.

Sources: the plan's research from 2026-10-27 and Anthropic's model table
(2026-10). Prices change - verify before the final paid run. Unknown routes
(local models, the fake provider) cost 0.
"""

import logging

from docrag.llm.types import Usage

log = logging.getLogger(__name__)

USD_TO_EUR = 0.89381  # rate from the plan, 2026-10-27

PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic:claude-haiku-5-5": (0.10, 0.50),
    "anthropic:claude-sonnet-5-5": (2.00, 10.00),
    "anthropic:claude-opus-5-5": (4.00, 20.00),
    "openai:gpt-5.6-luna": (0.10, 0.60),
    "mistral:mistral-small-4": (0.15, 0.60),
    "gemini:gemini-3.8-flash": (0.375, 1.875),
    "deepseek:deepseek-flash": (0.15, 0.60),
}
_FREE_PROVIDERS = ("ollama:", "fake:")
_warned: set[str] = set()


def cost_eur(route: str, usage: Usage) -> float:
    if usage.cost_usd is not None:  # the provider told us what it billed
        return usage.cost_usd * USD_TO_EUR
    price = PRICES_USD_PER_MTOK.get(route)
    if price is None:
        if not route.startswith(_FREE_PROVIDERS) and route not in _warned:
            _warned.add(route)
            log.warning("no price for %s - counting it as 0 EUR", route)
        return 0.0
    usd = usage.input_tokens / 1e6 * price[0] + usage.output_tokens / 1e6 * price[1]
    return usd * USD_TO_EUR
