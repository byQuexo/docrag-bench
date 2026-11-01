"""Render retrieved chunks as the LLM context, and load versioned prompts."""

from functools import lru_cache
from importlib import resources

from docrag.shared.schema import Hit


def render_context(hits: list[Hit]) -> str:
    """One block per chunk: ``[doc3:p2:c1] text``.

    The ID is the citation handle the LLM must copy. Newlines inside a chunk are
    flattened so each chunk starts on its own "[id]" line.
    """
    lines = []
    for hit in hits:
        c = hit.chunk
        section = " > ".join(c.section_path)
        pages = str(c.page_start)
        if c.page_end != c.page_start:
            pages += f"-{c.page_end}"
        header = f"(page {pages}, {c.type}" + (
            f", section: {section})" if section else ")"
        )
        body = (
            " ".join(c.text.split()) if c.type != "table" else c.text.replace("\n", " ")
        )
        lines.append(f"[{c.id}] {header} {body}")
    return "\n\n".join(lines)


@lru_cache
def load_prompt(version: str) -> str:
    """Prompts are files (prompts/answer_v1.txt), versioned by name.

    The version is part of every cache key through the prompt text itself, so
    editing a prompt = new file name = old cached answers are not reused.
    """
    # LEARN: importlib.resources reads files shipped INSIDE the package, so it
    # works the same from the repo, an installed wheel or the Docker image -
    # unlike paths relative to the current working directory.
    return (resources.files("docrag.prompts") / f"{version}.txt").read_text(
        encoding="utf-8"
    )
