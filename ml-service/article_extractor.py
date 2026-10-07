"""Article text extraction and passage splitting.

Downloads an article URL and extracts readable paragraphs, stripping
navigation, ads, footers, and cookie text.  Provides passage-level
splitting for NLI scoring.

WHY PASSAGE SELECTION MATTERS:
NLI only ever sees the handful of passages this module returns. Anything it
drops cannot support or contradict anything, no matter how decisive it is.
That makes ``extract_passages`` the narrowest point in the whole pipeline —
and it used to take the article's *first* few sentences, which in news
writing are scene-setting. A story reporting "the prime minister resigned,
effective immediately" in its eighth sentence was handed to NLI as six
sentences about the weather and the security cordon, scored neutral, and
filed as coverage that does not address the claim.
"""

from __future__ import annotations

import re
import time
from html import unescape
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from passage_retriever import reciprocal_rank_fusion, semantic_order

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


# Elements with no closing tag. Pushing them onto the stack desynchronises
# it: the matching </p> pops the void element instead, leaving "p" on the
# stack for the rest of the document.
_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})


# Elements whose contents are never prose. Their text is markup, code or
# styling that happens to be readable, and it is worse than useless as
# evidence: it is attacker-influenced text presented as publisher text.
_NON_PROSE_ELEMENTS = frozenset({
    "script", "style", "noscript", "template", "svg", "iframe", "code", "pre",
})


class _ArticleParser(HTMLParser):
    """Extracts title and readable paragraphs from raw HTML."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.paragraphs: list[str] = []
        self._tag_stack: list[str] = []
        self._title_parts: list[str] = []
        self._paragraph_parts: list[str] = []
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_ELEMENTS:
            return
        self._tag_stack.append(tag)
        if tag == "p":
            self._paragraph_parts = []
        if tag in {"h2", "h3"}:
            self._heading_parts = []

    def handle_endtag(self, tag):
        if tag in _VOID_ELEMENTS:
            return
        if tag == "title":
            self.title = " ".join(self._title_parts).strip()
        if tag == "p":
            paragraph = " ".join(self._paragraph_parts).strip()
            if len(paragraph.split()) >= 8:
                self.paragraphs.append(paragraph)
            self._paragraph_parts = []
        if tag in {"h2", "h3"}:
            heading = " ".join(self._heading_parts).strip()
            if len(heading.split()) >= 3:
                self.paragraphs.append(heading)
            self._heading_parts = []
        # Unwind to the matching open tag. Real-world HTML routinely omits
        # closing tags; popping blindly leaves the stack permanently wrong.
        if tag in self._tag_stack:
            while self._tag_stack:
                if self._tag_stack.pop() == tag:
                    break

    def handle_data(self, data):
        # HTMLParser hands back the body of <script> and <style> as ordinary
        # character data. Inside a <p> — where ad slots, embeds and analytics
        # beacons routinely sit — that meant page JavaScript was appended to
        # the article text and classified as something the article said. A
        # config blob containing "Google banned in all US cities" reached NLI
        # as a sentence from the publisher.
        if any(tag in _NON_PROSE_ELEMENTS for tag in self._tag_stack):
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self._title_parts.append(text)
        elif self._tag_stack and self._tag_stack[-1] in {"h2", "h3"}:
            self._heading_parts.append(text)
        elif "p" in self._tag_stack:
            self._paragraph_parts.append(text)


# Enough for any article; a cap is needed because search results sometimes
# point at very large pages, and the whole body is read into memory.
MAX_ARTICLE_BYTES = 2_000_000


def _fetch_html(url: str, timeout: int = 6, retries: int = 0) -> str:
    """Fetch an article page. No retry by default.

    This runs once per candidate document (up to 8 per claim). A news site
    that blocks or stalls a scraper on the first attempt almost never
    succeeds on a retry, so retrying just multiplies the wall-clock cost of
    a failure across every candidate.

    Non-HTML responses raise rather than being decoded: search results do
    sometimes point at a PDF or an image, and decoding those as text yields
    binary noise that would be handed to NLI as if it were prose.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                content_type = (response.headers.get_content_type() or "").lower()
                if content_type and not (
                    content_type.startswith("text/")
                    or content_type.endswith("+xml")
                    or content_type == "application/xhtml+xml"
                ):
                    raise ValueError(f"not an HTML document: {content_type}")
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read(MAX_ARTICLE_BYTES).decode(charset, errors="ignore")
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.0)
    raise last_error


# Everything a <script> or <style> encloses, removed before the fallback reads
# the page as flat text. Without this the fallback would reintroduce exactly
# the leak the parser was fixed to prevent.
_NON_PROSE_BLOCK_RE = re.compile(
    r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")

# Block-level elements become line breaks before the markup is thrown away.
# Without a boundary the flat text runs a nav bar straight into the lede —
# "Home World Business Sport The minister resigned on Tuesday." — which is one
# unsplittable sentence, so the navigation could not be filtered off the front
# of the story.
_BLOCK_BOUNDARY_RE = re.compile(
    r"</?(?:div|p|li|ul|ol|section|article|aside|nav|footer|header|main|"
    r"h[1-6]|td|tr|table|br|blockquote|figcaption|form|button)\b[^>]*>",
    re.IGNORECASE,
)

# A block of flat text is treated as a sentence of article prose only if it is
# this long and ends like a sentence. Navigation items, bylines, tags and
# button labels are short and unpunctuated; this is what separates them from
# reporting once the markup that distinguished them is gone.
_FALLBACK_MIN_WORDS = 8


def _text_without_markup(html: str) -> str:
    """Readable text from a page whose body is not in <p> elements.

    Not the normal path, and deliberately not the first choice: <p> carries
    the publisher's own judgement about what is a paragraph, and throwing the
    markup away loses it. But a page that builds its body from <div> — AMP
    templates and several large CMSs do — yielded ZERO paragraphs and
    contributed nothing at all, no matter what it reported. Falling back to
    flat text turns "this publisher never counts" into "this publisher counts,
    with more noise", and the noise is then filtered like any other text.
    """
    stripped = _NON_PROSE_BLOCK_RE.sub(" ", html)
    stripped = _BLOCK_BOUNDARY_RE.sub("\n", stripped)
    text = unescape(_TAG_RE.sub(" ", stripped))

    prose: list[str] = []
    for block in text.split("\n"):
        block = re.sub(r"[ \t]+", " ", block).strip()
        if not block:
            continue
        for sentence in split_sentences(block):
            if (len(sentence.split()) >= _FALLBACK_MIN_WORDS
                    and sentence.rstrip()[-1:] in ".!?\u201d\""
                    and not is_boilerplate(sentence)):
                prose.append(sentence)
    return " ".join(prose)


def extract_article(url: str, timeout: int = 6) -> tuple[str, str]:
    """Download a URL and return (title, full_text)."""
    html = _fetch_html(url, timeout=timeout)
    parser = _ArticleParser()
    parser.feed(html)
    text = " ".join(parser.paragraphs)
    if not text:
        text = _text_without_markup(html)
    return parser.title, text


# Abbreviations whose full stop does not end a sentence. Splitting on them
# silently truncates the text: "The U.S. government banned Google" split after
# "U." and, with the two-word fragment discarded, became "government banned
# Google" — the claim's subject deleted.
#
# Exported because claim_verifier splits *user input* the same way and needs
# the same protection. It had its own splitter without it, so the protection
# existed in the codebase but not where it mattered most.
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Sen", "Rep", "Gov",
    "Pres", "Gen", "Lt", "Col", "Capt", "Rev", "Hon",
    "vs", "etc", "approx", "est", "govt", "dept", "inc", "ltd", "co", "corp",
    "no", "fig", "al",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct",
    "Nov", "Dec",
    r"U\.S", r"U\.K", r"U\.N", r"E\.U", r"D\.C",
)
_ABBREVIATION_RE = re.compile(
    r"\b(" + "|".join(_ABBREVIATIONS) + r")\.", re.IGNORECASE
)
_DOT_PLACEHOLDER = "<DOT>"


def protect_abbreviations(text: str) -> str:
    """Hide the full stops in abbreviations so they can't end a sentence."""
    return _ABBREVIATION_RE.sub(
        lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), text
    )


def restore_abbreviations(text: str) -> str:
    """Undo :func:`protect_abbreviations`."""
    return text.replace(_DOT_PLACEHOLDER, ".")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations like 'Mr.' and 'U.S.'."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    protected = protect_abbreviations(cleaned)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(\[])', protected)
    return [
        restore_abbreviations(part).strip()
        for part in parts
        if len(part.split()) >= 5
    ]


# Words too common to indicate that a sentence is about the claim.
_PASSAGE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "from", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "do", "does", "did", "will",
    "would", "can", "could", "may", "might", "must", "should", "that",
    "this", "these", "those", "it", "its", "they", "them", "he", "she",
    "his", "her", "their", "we", "you", "not", "no", "than", "then", "so",
    "said", "says", "after", "before", "over", "into", "about", "more",
    "also", "who", "which", "when", "where", "what", "how", "why",
})


# Site furniture that survives HTML extraction: the subscription pitch, the
# consent notice, the newsletter box, the social footer. None of it is what the
# article says, and on a paywalled page it is very nearly ALL the page ships.
#
# These are PHRASES, never bare words. "subscription" alone would delete the
# real content of any article about streaming prices; "subscribe today to
# continue reading" cannot appear in a news sentence by accident. That, plus
# the claim-overlap escape in is_boilerplate(), is what keeps the filter from
# eating the story it is meant to clean up.
_BOILERPLATE_PHRASES = (
    # Paywall and subscription
    "continue reading", "already a subscriber", "subscribe today",
    "subscribe now", "your subscription", "unlimited digital access",
    "cancel anytime", "free trial", "sign in to read", "create an account",
    "support independent journalism", "choose a plan", "a week for",
    "this article is for subscribers", "become a member",
    # Consent and privacy
    "we use cookies", "accept all cookies", "cookie policy", "privacy policy",
    "terms of service", "manage your preferences", "consent to the use",
    # Newsletter and social furniture
    "sign up for our newsletter", "sign up for the newsletter",
    "follow us on", "share this article", "download our app",
    "get the latest news delivered", "newsletter signup",
    # Legal and navigation footers
    "all rights reserved", "advertisement", "read more:", "related articles",
    "photograph:", "image credit", "copyright ",
)


def is_boilerplate(sentence: str, claim_words: set[str] | None = None) -> bool:
    """True when a sentence is site furniture rather than article content.

    ``claim_words`` is an escape hatch, not an optimisation. A claim about
    streaming prices makes "your subscription will renew automatically" a
    sentence genuinely worth reading, and a claim about data protection does
    the same for a cookie notice. A sentence sharing vocabulary with the claim
    is therefore never discarded, whatever phrase it matched.
    """
    lowered = (sentence or "").lower()
    if not any(phrase in lowered for phrase in _BOILERPLATE_PHRASES):
        return False
    if claim_words and (claim_words & _content_words(sentence)):
        return False
    return True


def strip_boilerplate(text: str, claim: str = "") -> str:
    """``text`` with its site furniture removed, sentence by sentence."""
    if not text:
        return text
    claim_words = _content_words(claim)
    kept = [s for s in split_sentences(text) if not is_boilerplate(s, claim_words)]
    return " ".join(kept)


def _content_words(text: str) -> set[str]:
    """Lowercased words that carry topic, for overlap scoring."""
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _PASSAGE_STOPWORDS
    }


def _lexical_order(sentences: list[str], claim_words: set[str]) -> tuple[list[int], list[int]]:
    """(every index in lexical order, the indices that share any claim word).

    Sort by overlap, then by original position, so ties keep article order and
    the lede still wins when nothing matches.
    """
    overlaps = [len(claim_words & _content_words(s)) for s in sentences]
    order = sorted(range(len(sentences)), key=lambda i: (-overlaps[i], i))
    return order, [i for i in order if overlaps[i] > 0]


def extract_passages(
    title: str,
    snippet: str,
    full_text: str,
    max_passages: int = 8,
    claim: str = "",
    ranking_info: dict | None = None,
) -> list[str]:
    """Return the passages most worth showing NLI for this claim.

    The title and snippet always come first — they are curated summaries of
    what the article is about. The remaining slots go to the article
    sentences that best match ``claim``: by content-word overlap, and — when
    dense ranking is available — by meaning, the two merged with reciprocal
    rank fusion (see ``passage_retriever``).

    WHY DENSE RANKING WAS ADDED:
    Overlap cannot see a paraphrase, and debunks are paraphrases. "Washington
    has not prohibited the search giant anywhere in the country" shares no
    word with "The United States banned Google across all its cities", so it
    scored zero and lost its slot to any sentence that mentioned Google. The
    embedding ranking nominates it; fusion keeps exact name and number
    matches competitive, which embeddings blur.

    Only sentences one of the two signals actually NOMINATED are fused —
    those with some word overlap, and those above the similarity floor. The
    rest follow in lexical order, which for them is article order. If dense
    ranking is disabled, failed (``None``) or found nothing close (``[]``),
    the result is exactly the lexical-only behaviour.

    ``ranking_info``, when given, receives ``{"ranking": "hybrid" | "lexical"}``
    so the pipeline can report how often the dense ranking actually decided.

    WHY IT RANKS RATHER THAN TRUNCATES:
    This used to return the article's first ``max_passages`` sentences. News
    writing opens with scene-setting and states the specific fact several
    paragraphs down, so the decisive sentence routinely fell outside the
    window: a story reporting a resignation in its eighth sentence reached
    NLI as six sentences about the weather and the security cordon, was
    scored neutral, and was filed as coverage that does not address the
    claim. For a high-salience claim that is worse than unhelpful — enough
    such articles and the aggregator concludes nobody reported an event that
    was, in fact, reported.

    With no ``claim`` (or a claim sharing no vocabulary with the article)
    every sentence scores zero and the original document order is preserved,
    so behaviour degrades exactly to what it was before.
    """
    passages: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        key = re.sub(r"\W+", " ", (text or "").lower()).strip()
        # Google News descriptions frequently restate the headline verbatim;
        # duplicated passages waste NLI slots on the same sentence.
        if key and key not in seen:
            seen.add(key)
            passages.append(text)

    add(title)
    add(snippet)

    sentences = split_sentences(full_text)
    claim_words = _content_words(claim)
    order, lexical_hits = _lexical_order(sentences, claim_words)

    ranking = "lexical"
    dense = semantic_order(claim, sentences) if claim and sentences else None
    if dense:
        fused = reciprocal_rank_fusion(dense, lexical_hits)
        nominated = set(fused)
        order = fused + [i for i in order if i not in nominated]
        ranking = "hybrid"
    if ranking_info is not None:
        ranking_info["ranking"] = ranking

    for sentence in (sentences[i] for i in order):
        if len(passages) >= max_passages:
            break
        if is_boilerplate(sentence, claim_words):
            continue
        add(sentence)

    return passages[:max_passages]
