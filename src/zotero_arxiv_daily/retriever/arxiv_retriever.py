from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import re

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
ARXIV_ID_BATCH_SIZE = 5
ARXIV_REQUEST_INTERVAL_SECONDS = 4
ARXIV_RETRY_BACKOFF_SECONDS = (15, 30)
ARXIV_RETRYABLE_HTTP_STATUSES = {429, 503}
ARXIV_DEFAULT_FETCH_MULTIPLIER = 4


@dataclass
class _RssAuthor:
    name: str


@dataclass
class _RssArxivPaper:
    title: str
    authors: list[_RssAuthor]
    summary: str
    pdf_url: str | None
    entry_id: str
    paper_id: str

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self.paper_id}"


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _get_feed_title(feed: Any) -> str:
    feed_metadata = getattr(feed, "feed", None) or {}
    if hasattr(feed_metadata, "get"):
        return str(feed_metadata.get("title", "") or "")
    return str(getattr(feed_metadata, "title", "") or "")


def _get_http_status(exc: Exception) -> int | None:
    for attr in ("status", "status_code", "code"):
        status = getattr(exc, attr, None)
        if status is None:
            continue
        try:
            return int(status)
        except (TypeError, ValueError):
            return None
    return None


def _extract_arxiv_id(entry: Any) -> str:
    return str(entry.id).removeprefix("oai:arXiv.org:")


def _entry_link(entry: Any, paper_id: str) -> str:
    link = entry.get("link") if hasattr(entry, "get") else None
    if link:
        return str(link)
    return f"https://arxiv.org/abs/{paper_id}"


def _clean_rss_summary(summary: str) -> str:
    summary = re.sub(
        r"^\s*arXiv:\S+\s+Announce Type:\s+\w+\s*Abstract:\s*",
        "",
        summary,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return summary.strip()


def _keyword_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def _entry_search_text(entry: Any) -> str:
    title = str(entry.get("title", "") if hasattr(entry, "get") else getattr(entry, "title", ""))
    summary = str(entry.get("summary", "") if hasattr(entry, "get") else getattr(entry, "summary", ""))
    return f"{title}\n{_clean_rss_summary(summary)}".lower()


def _entry_matches_keywords(entry: Any, include_keywords: list[str], exclude_keywords: list[str]) -> bool:
    text = _entry_search_text(entry)
    if include_keywords and not any(keyword.lower() in text for keyword in include_keywords):
        return False
    if exclude_keywords and any(keyword.lower() in text for keyword in exclude_keywords):
        return False
    return True


def _rss_entry_to_paper(entry: Any) -> _RssArxivPaper:
    paper_id = _extract_arxiv_id(entry)
    title = str(entry.get("title", "") if hasattr(entry, "get") else getattr(entry, "title", ""))
    summary = str(entry.get("summary", "") if hasattr(entry, "get") else getattr(entry, "summary", ""))
    creators = ""
    if hasattr(entry, "get"):
        creators = entry.get("dc_creator") or entry.get("author") or ""
    authors = [_RssAuthor(name.strip()) for name in str(creators).split(",") if name.strip()]
    entry_id = _entry_link(entry, paper_id)
    return _RssArxivPaper(
        title=title.strip(),
        authors=authors,
        summary=_clean_rss_summary(summary),
        pdf_url=f"https://arxiv.org/pdf/{paper_id}" if paper_id else None,
        entry_id=entry_id,
        paper_id=paper_id,
    )


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _get_max_feed_papers(self) -> int | None:
        configured_limit = self.config.source.arxiv.get("max_fetch_papers")
        if configured_limit is not None:
            return int(configured_limit)

        max_paper_num = self.config.executor.get("max_paper_num")
        if max_paper_num is None:
            return None
        return int(max_paper_num) * ARXIV_DEFAULT_FETCH_MULTIPLIER

    def _retrieve_raw_papers(self) -> list[ArxivResult | _RssArxivPaper]:
        client = arxiv.Client(num_retries=0, delay_seconds=ARXIV_REQUEST_INTERVAL_SECONDS)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in _get_feed_title(feed):
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        include_keywords = _keyword_list(self.config.source.arxiv.get("include_keywords"))
        exclude_keywords = _keyword_list(self.config.source.arxiv.get("exclude_keywords"))
        feed_entries = [
            entry
            for entry in getattr(feed, "entries", []) or []
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if include_keywords or exclude_keywords:
            before_keyword_filter = len(feed_entries)
            feed_entries = [
                entry for entry in feed_entries
                if _entry_matches_keywords(entry, include_keywords, exclude_keywords)
            ]
            logger.info(
                f"Filtered arXiv RSS entries by keywords from {before_keyword_filter} to {len(feed_entries)}"
            )
        max_feed_papers = self._get_max_feed_papers()
        if max_feed_papers is not None and len(feed_entries) > max_feed_papers:
            logger.info(
                f"Limiting arXiv RSS entries from {len(feed_entries)} to {max_feed_papers} "
                f"before API enrichment"
            )
            feed_entries = feed_entries[:max_feed_papers]

        fallback_by_id = {_extract_arxiv_id(entry): _rss_entry_to_paper(entry) for entry in feed_entries}
        all_paper_ids = [
            _extract_arxiv_id(i)
            for i in feed_entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        logger.info(
            f"Fetching {len(all_paper_ids)} arXiv papers in batches of "
            f"{ARXIV_ID_BATCH_SIZE} with at least {ARXIV_REQUEST_INTERVAL_SECONDS}s between requests"
        )
        bar = tqdm(total=len(all_paper_ids))
        total_batches = (len(all_paper_ids) + ARXIV_ID_BATCH_SIZE - 1) // ARXIV_ID_BATCH_SIZE
        if all_paper_ids:
            sleep(ARXIV_REQUEST_INTERVAL_SECONDS)
        for batch_index, i in enumerate(range(0, len(all_paper_ids), ARXIV_ID_BATCH_SIZE)):
            if batch_index > 0:
                sleep(ARXIV_REQUEST_INTERVAL_SECONDS)

            batch_ids = all_paper_ids[i:i + ARXIV_ID_BATCH_SIZE]
            search = arxiv.Search(id_list=batch_ids)
            for attempt in range(len(ARXIV_RETRY_BACKOFF_SECONDS) + 1):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch_ids))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    status = _get_http_status(exc)
                    can_retry = (
                        status in ARXIV_RETRYABLE_HTTP_STATUSES
                        and attempt < len(ARXIV_RETRY_BACKOFF_SECONDS)
                    )
                    if can_retry:
                        wait = ARXIV_RETRY_BACKOFF_SECONDS[attempt]
                        logger.warning(
                            f"arXiv API HTTP {status} on batch {batch_index + 1}/{total_batches}, "
                            f"retry {attempt + 1}/{len(ARXIV_RETRY_BACKOFF_SECONDS)} in {wait}s"
                        )
                        sleep(wait)
                        continue

                    logger.warning(
                        f"Falling back to RSS data for arXiv batch {batch_index + 1}/{total_batches} "
                        f"after HTTP {status or 'unknown'} error: {exc}"
                    )
                    raw_papers.extend(fallback_by_id[paper_id] for paper_id in batch_ids if paper_id in fallback_by_id)
                    bar.update(len(batch_ids))
                    break
                except Exception as exc:
                    logger.warning(
                        f"Falling back to RSS data for arXiv batch {batch_index + 1}/{total_batches} "
                        f"after unexpected error: {exc}"
                    )
                    raw_papers.extend(fallback_by_id[paper_id] for paper_id in batch_ids if paper_id in fallback_by_id)
                    bar.update(len(batch_ids))
                    break
        bar.close()

        if len(raw_papers) < len(all_paper_ids):
            logger.warning(f"Fetched {len(raw_papers)}/{len(all_paper_ids)} arXiv papers; continuing with partial results")

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult | _RssArxivPaper) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        if isinstance(raw_paper, _RssArxivPaper):
            full_text = None
        else:
            full_text = extract_text_from_tar(raw_paper)
            if full_text is None:
                full_text = extract_text_from_html(raw_paper)
            if full_text is None:
                full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
