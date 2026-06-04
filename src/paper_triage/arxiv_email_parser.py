from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger


@dataclass(frozen=True)
class EmailPaper:
    title: str
    abstract: str
    arxiv_url: str


_ARXIV_URL_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/[^\s\"'<>]+", re.I)
_ARXIV_PATH_RE = re.compile(r"^/(?:abs|pdf)/[^\s\"'<>]+", re.I)
_LABEL_RE = re.compile(r"^(abstract|摘要|tldr|tl;dr|summary|简介)\s*[:：]\s*(.*)$", re.I)
_STOP_LABEL_RE = re.compile(
    r"^(relevance|score|authors?|pdf|arxiv|affiliations?|相关性|作者|链接)\s*[:：]?",
    re.I,
)


def normalize_arxiv_url(url: str) -> str:
    """Return an arXiv abs URL when the input is an arXiv abs/pdf URL."""
    cleaned = unescape(url or "").strip()
    if not cleaned:
        return ""

    if cleaned.startswith("//"):
        cleaned = "https:" + cleaned
    elif cleaned.startswith("/abs/") or cleaned.startswith("/pdf/"):
        cleaned = "https://arxiv.org" + cleaned

    parsed = urlparse(cleaned)
    if "arxiv.org" not in parsed.netloc.lower():
        return cleaned

    path = parsed.path.strip("/")
    if path.startswith("abs/"):
        paper_id = path.removeprefix("abs/")
    elif path.startswith("pdf/"):
        paper_id = path.removeprefix("pdf/")
        paper_id = re.sub(r"\.pdf$", "", paper_id, flags=re.I)
    else:
        return cleaned

    return f"https://arxiv.org/abs/{paper_id}"


def parse_arxiv_email(html: str, max_papers: int = 5) -> list[EmailPaper]:
    """Parse a zotero-arxiv-daily HTML email into paper records."""
    soup = BeautifulSoup(html or "", "html.parser")
    papers: list[EmailPaper] = []

    containers = _paper_containers(soup)
    logger.info("Found {} candidate paper blocks in email", len(containers))
    for container in containers:
        title = _extract_title(container)
        link = _extract_arxiv_url(container)
        abstract = _extract_abstract(container)

        if not title or not link:
            logger.debug("Skipping candidate block without title or arXiv link")
            continue

        papers.append(EmailPaper(title=title, abstract=abstract, arxiv_url=link))
        if len(papers) >= max_papers:
            break

    if papers:
        return papers

    # Fallback for simpler plain-text digests that are not table based.
    text = soup.get_text("\n")
    for url in _extract_urls_from_text(text):
        nearby_title = _title_before_url(text, url)
        if nearby_title:
            papers.append(EmailPaper(title=nearby_title, abstract="", arxiv_url=normalize_arxiv_url(url)))
        if len(papers) >= max_papers:
            break
    return papers


def _paper_containers(soup: BeautifulSoup) -> list:
    tables = [
        table
        for table in soup.find_all("table")
        if _extract_arxiv_url(table) and "No Papers Today" not in table.get_text(" ")
    ]
    if tables:
        return tables

    blocks = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "arxiv.org" in href:
            blocks.append(link.find_parent(["article", "section", "div", "li"]) or link.parent)
    return blocks


def _extract_arxiv_url(container) -> str:
    for link in container.find_all("a", href=True):
        href = link["href"].strip()
        if "arxiv.org" in href or _ARXIV_PATH_RE.match(href):
            return normalize_arxiv_url(href)

    text = container.get_text(" ")
    urls = _extract_urls_from_text(text)
    return normalize_arxiv_url(urls[0]) if urls else ""


def _extract_urls_from_text(text: str) -> list[str]:
    urls = _ARXIV_URL_RE.findall(text or "")
    urls.extend(f"https://arxiv.org{path}" for path in _ARXIV_PATH_RE.findall(text or ""))
    return urls


def _extract_title(container) -> str:
    for cell in container.find_all(["td", "h1", "h2", "h3", "strong", "b"]):
        text = _clean_text(cell.get_text(" "))
        if _looks_like_title(text):
            return text

    for line in _clean_text(container.get_text("\n")).splitlines():
        if _looks_like_title(line):
            return line
    return ""


def _looks_like_title(text: str) -> bool:
    if not text or len(text) < 8:
        return False
    lowered = text.lower()
    if lowered in {"pdf", "abs", "arxiv"}:
        return False
    if _LABEL_RE.match(text) or _STOP_LABEL_RE.match(text):
        return False
    if text.count(" ") < 1:
        return False
    return True


def _extract_abstract(container) -> str:
    lines = [_clean_text(line) for line in container.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    for index, line in enumerate(lines):
        match = _LABEL_RE.match(line)
        if not match:
            continue

        parts = []
        if match.group(2):
            parts.append(match.group(2))

        for next_line in lines[index + 1 :]:
            if _LABEL_RE.match(next_line) or _STOP_LABEL_RE.match(next_line):
                break
            if next_line.lower() in {"pdf", "abs", "arxiv"}:
                break
            parts.append(next_line)

        abstract = _clean_text(" ".join(parts))
        if abstract:
            return abstract

    return ""


def _title_before_url(text: str, url: str) -> str:
    before = text.split(url, 1)[0]
    lines = [_clean_text(line) for line in before.splitlines()]
    for line in reversed(lines):
        if _looks_like_title(line):
            return line
    return ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text or "")).strip()

