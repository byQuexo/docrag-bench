"""Print the latest trace in Langfuse as a tree, for auditing the instrumentation.

    uv run python scripts/langfuse_trace_audit.py [--trace-id ID] [--minutes 120]

Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST through Settings
and queries the Observations API v2 (the traces endpoints are legacy). Shows,
per observation: type, name, model, tokens, cost, and the first characters of
input/output, so gaps (missing types, empty trace I/O, dropped attributes) are
visible at a glance.
"""

import argparse
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from docrag.shared.settings import get_settings

FIELDS = "core,basic,io,model,usage,metadata,trace_context"


def fetch(client: httpx.Client, params: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = {**params, **({"cursor": cursor} if cursor else {})}
        r = client.get("/api/public/v2/observations", params=page)
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("data", []))
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor:
            return out


def short(value: Any, n: int = 70) -> str:
    text = " ".join(str(value).split()) if value not in (None, "") else "-"
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-id")
    parser.add_argument("--minutes", type=int, default=120)
    args = parser.parse_args()
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        raise SystemExit("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    since = datetime.now(UTC) - timedelta(minutes=args.minutes)
    base = {"fromStartTime": since.isoformat(), "limit": 1000, "fields": FIELDS}
    auth = (s.langfuse_public_key, s.langfuse_secret_key.get_secret_value())
    with httpx.Client(base_url=s.langfuse_host, auth=auth, timeout=30) as client:
        trace_id = args.trace_id
        if trace_id is None:
            roots = fetch(client, {**base, "isRootObservation": "true", "limit": 1})
            if not roots:
                raise SystemExit(f"no traces in the last {args.minutes} minutes")
            trace_id = roots[0]["traceId"]
        obs = fetch(client, {**base, "traceId": trace_id})

    children: dict[str | None, list[dict[str, Any]]] = {}
    for o in sorted(obs, key=lambda o: o["startTime"]):
        children.setdefault(o.get("parentObservationId"), []).append(o)
    root = children.get(None, [{}])[0]
    print(f"trace {trace_id}  name={root.get('traceName') or root.get('name')!r}")
    print(f"  session={root.get('sessionId')}  env={root.get('environment')}")
    print(f"  tags={root.get('tags')}  release={root.get('release')}")
    print(f"  total cost={sum(float(o.get('totalCost') or 0) for o in obs):.6f} USD")

    def show(o: dict[str, Any], depth: int) -> None:
        pad = "  " * depth
        extra = ""
        if o.get("type") == "GENERATION":
            extra = (
                f" model={o.get('model')} usage={o.get('usageDetails')} "
                f"cost={o.get('totalCost')}"
            )
        print(f"{pad}- [{o.get('type')}] {o.get('name')}{extra}")
        print(f"{pad}    in : {short(o.get('input'))}")
        print(f"{pad}    out: {short(o.get('output'))}")
        for child in children.get(o["id"], []):
            show(child, depth + 1)

    for top in children.get(None, []):
        show(top, 0)


if __name__ == "__main__":
    main()
