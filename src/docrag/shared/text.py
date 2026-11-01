"""Text normalization used by Verify (quote matching) and BM25 (tokenizing).

One place, so "the quote is in the chunk" and "the term is in the chunk" use the
same notion of equal text.
"""

import re
import unicodedata

# Typographic characters that PDFs love and LLMs "fix" when quoting.
_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
        "­": "",  # soft hyphen
    }
)
_WS = re.compile(r"\s+")
# A word broken across lines in a PDF: "infor-\nmation" -> "information".
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes, rejoin hyphenated line breaks, squash spaces.

    >>> normalize("The  “Revenue”\\nwas infor-\\nmative")
    'the "revenue" was informative'
    """
    # NFKC turns ligatures common in PDFs ("ﬁ", one char) into plain letters ("fi").
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.translate(_TRANSLATE).lower()
    return _WS.sub(" ", text).strip()


# Short English stopword list. Dropping these keeps BM25 sparse vectors small and
# stops "the" from matching everything. Known limitation: English only.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with what which who whom how why when where does did do
    can could would should not no than then there these those their they them
    he she his her we our you your i me my""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords. Numbers like 1,500 or 3.3 stay whole."""
    return [t for t in _TOKEN.findall(normalize(text)) if t not in STOPWORDS]
