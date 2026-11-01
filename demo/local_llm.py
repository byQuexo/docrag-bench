"""An open answer model running inside the Space (Hugging Face ZeroGPU).

Imported only when DOCRAG_DEMO_LOCAL_MODEL is set (e.g. "Qwen/Qwen3.5-9B").
ZeroGPU rules (https://huggingface.co/docs/hub/spaces-zerogpu): load the model at
import time and move it to "cuda" there (ZeroGPU emulates CUDA outside GPU
functions); only the function decorated with @spaces.GPU gets a real GPU, per
call, billed to the visitor's daily quota. Outside a ZeroGPU Space the decorator
does nothing, so this also runs locally (CPU, slowly) for testing.

It implements the same Provider interface as the API providers, so the proxy,
Verify and the UI don't know the difference.
"""

import asyncio
import json
import os
from typing import Any

import spaces
import torch
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from docrag.llm.base import Provider
from docrag.llm.types import ChatCompletion, Message, Usage, json_schema_for

MODEL_ID = os.environ["DOCRAG_DEMO_LOCAL_MODEL"]
# "auto": GPU if torch sees one (ZeroGPU emulates it at startup), else CPU.
_device = os.environ.get("DOCRAG_DEMO_DEVICE", "auto")
DEVICE = (
    ("cuda" if torch.cuda.is_available() else "cpu") if _device == "auto" else _device
)
MAX_NEW_TOKENS = int(os.environ.get("DOCRAG_DEMO_MAX_NEW_TOKENS", "1024"))

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
MODEL.eval()


@spaces.GPU(duration=90)  # type: ignore[untyped-decorator]  # 10 chunks + JSON fit
def generate(prompt: str, max_new_tokens: int) -> tuple[str, int, int]:
    """Greedy decoding. Plain str/int in and out: ZeroGPU runs this in a worker."""
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.inference_mode():
        out = MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, inputs["input_ids"].shape[1] :]
    text = TOKENIZER.decode(new, skip_special_tokens=True)
    return text, int(inputs["input_ids"].shape[1]), int(new.shape[0])


class LocalProvider(Provider):
    name = "local"

    def to_native(
        self,
        model: str,
        messages: list[Message],
        response_format: type[BaseModel] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        chat = [{"role": m.role, "content": m.text} for m in messages]
        if response_format is not None:
            # API providers get the schema as a parameter; a local model needs
            # it in the prompt. Appended to the (first) system message.
            schema = json.dumps(json_schema_for(response_format))
            note = (
                f"\n\nReply with one JSON object matching this JSON schema:\n{schema}"
            )
            if chat and chat[0]["role"] == "system":
                chat[0]["content"] += note
            else:
                chat.insert(0, {"role": "system", "content": note.strip()})
        # enable_thinking=False: Qwen models would otherwise spend the token
        # budget on hidden reasoning (seen in the benchmark as empty answers).
        prompt = TOKENIZER.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return {"model": model, "prompt": prompt}

    async def _call(self, native: dict[str, Any]) -> Any:
        return await asyncio.to_thread(generate, native["prompt"], MAX_NEW_TOKENS)

    def from_native(self, model: str, raw: Any) -> ChatCompletion:
        text, n_in, n_out = raw
        return ChatCompletion(
            model=model,
            content=text,
            usage=Usage(input_tokens=n_in, output_tokens=n_out, cost_usd=0.0),
        )
