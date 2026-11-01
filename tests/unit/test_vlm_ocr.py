"""Vision-LLM OCR: image messages in both provider formats, and the VlmOcrEngine."""

import base64
from pathlib import Path

from pydantic import BaseModel

from docrag.ingest.ocr import ocr_blocks
from docrag.ingest.vlm_ocr import VlmOcrEngine
from docrag.llm.anthropic_provider import AnthropicProvider
from docrag.llm.fake import FakeProvider
from docrag.llm.openai_compat import OpenAICompatProvider
from docrag.llm.proxy import LLMProxy
from docrag.llm.types import ImagePart, Message, TextPart
from docrag.shared.cache import Cache

SCANNED = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scanned.pdf"
PNG = b"\x89PNG\r\n\x1a\nfake"


def _image_message() -> Message:
    return Message(
        role="user",
        content=[ImagePart.from_png(PNG), TextPart(text="Transcribe this page.")],
    )


def test_message_text_ignores_images() -> None:
    assert _image_message().text == "Transcribe this page."
    assert Message(role="user", content="plain").text == "plain"


def test_openai_format_uses_image_url_data_urls() -> None:
    provider = OpenAICompatProvider("x", "http://example.invalid/v1")
    body = provider.to_native("m", [_image_message()], None, None)
    image, text = body["messages"][0]["content"]
    assert image["type"] == "image_url"
    b64 = base64.b64encode(PNG).decode()
    assert image["image_url"]["url"] == f"data:image/png;base64,{b64}"
    assert text == {"type": "text", "text": "Transcribe this page."}


def test_anthropic_format_uses_base64_image_blocks() -> None:
    provider = AnthropicProvider(api_key="test")
    native = provider.to_native(
        "m", [Message(role="system", content="sys"), _image_message()], None, None
    )
    assert native["system"] == "sys"
    image, text = native["messages"][0]["content"]
    assert image["type"] == "image"
    assert image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == PNG
    assert text == {"type": "text", "text": "Transcribe this page."}


def _transcriber(messages: list[Message], schema: type[BaseModel] | None) -> str:
    # The fake "sees" the image only as bytes; prove it was sent.
    user = messages[-1]
    assert isinstance(user.content, list)
    assert isinstance(user.content[0], ImagePart)
    return "Revenue by region\nFigure: table Europe 41.0 45.5"


def test_vlm_engine_ocrs_every_page_and_marks_origin_vlm() -> None:
    provider = FakeProvider(_transcriber)
    engine = VlmOcrEngine(LLMProxy({"fake": provider}), "fake:vision", concurrency=2)
    try:
        blocks = ocr_blocks(SCANNED.read_bytes(), "scan", engine, None, first_order=0)
    finally:
        engine.close()
    assert [b.page for b in blocks] == [1, 2, 3]
    assert all(b.origin == "vlm" for b in blocks)
    assert "Figure: table Europe" in blocks[1].text
    assert len(provider.calls) == 3


def test_failed_pages_are_not_cached_and_retried_next_time() -> None:
    attempts = {"n": 0}

    def flaky(messages: list[Message], schema: type[BaseModel] | None) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("provider hiccup")
        return "page text"

    cache = Cache(":memory:")
    engine = VlmOcrEngine(
        LLMProxy({"fake": FakeProvider(flaky)}), "fake:vision", concurrency=1
    )
    try:
        first = ocr_blocks(SCANNED.read_bytes(), "scan", engine, cache, first_order=0)
        second = ocr_blocks(SCANNED.read_bytes(), "scan", engine, cache, first_order=0)
    finally:
        engine.close()
    assert len(first) == 2  # one page failed -> no block for it, nothing cached
    assert len(second) == 3  # retried; the other two came from the cache
    assert attempts["n"] == 4
