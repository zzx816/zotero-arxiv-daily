"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def _make_arxiv_feed(paper_count: int):
    entries = [
        feedparser.FeedParserDict(
            {
                "id": f"oai:arXiv.org:2601.{index:05d}v1",
                "arxiv_announce_type": "new",
                "title": f"Paper {index}",
                "summary": f"arXiv:2601.{index:05d}v1 Announce Type: new\nAbstract: Abstract {index}",
                "dc_creator": f"Author {index}",
                "link": f"https://arxiv.org/abs/2601.{index:05d}v1",
            }
        )
        for index in range(paper_count)
    ]
    return feedparser.FeedParserDict(
        {
            "feed": feedparser.FeedParserDict({"title": "cs.AI updates on arXiv.org"}),
            "entries": entries,
        }
    )


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_arxiv_retriever_uses_small_batches_and_request_delays(config, monkeypatch):
    feed = _make_arxiv_feed(12)
    sleep_values: list[float] = []
    client_kwargs: list[dict] = []
    requested_batches: list[list[str]] = []

    class FakeSearch:
        def __init__(self, id_list):
            self.id_list = list(id_list)

    class FakeClient:
        def __init__(self, **kwargs):
            client_kwargs.append(kwargs)

        def results(self, search):
            requested_batches.append(search.id_list)
            return iter([SimpleNamespace(title=paper_id) for paper_id in search.id_list])

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Search", FakeSearch)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "sleep", sleep_values.append)

    raw_papers = ArxivRetriever(config)._retrieve_raw_papers()

    assert len(raw_papers) == 12
    assert [len(batch) for batch in requested_batches] == [5, 5, 2]
    assert client_kwargs[0]["num_retries"] == 0
    assert client_kwargs[0]["delay_seconds"] == arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS
    assert sleep_values == [
        arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS,
        arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS,
        arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS,
    ]


def test_arxiv_retriever_retries_and_skips_failed_batch(config, monkeypatch):
    feed = _make_arxiv_feed(11)
    sleep_values: list[float] = []
    warnings: list[str] = []
    batch_calls: dict[tuple[str, ...], int] = {}

    class FakeHTTPError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status = status

    class FakeSearch:
        def __init__(self, id_list):
            self.id_list = list(id_list)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            batch_key = tuple(search.id_list)
            batch_calls[batch_key] = batch_calls.get(batch_key, 0) + 1
            if len(batch_key) == 5 and batch_key[0].endswith("00005v1"):
                raise FakeHTTPError(503)
            return iter([SimpleNamespace(title=paper_id) for paper_id in search.id_list])

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "HTTPError", FakeHTTPError)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Search", FakeSearch)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "sleep", sleep_values.append)
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(info=lambda _: None, warning=warnings.append))

    raw_papers = ArxivRetriever(config)._retrieve_raw_papers()
    failed_batch = tuple(entry.id.removeprefix("oai:arXiv.org:") for entry in feed.entries[5:10])

    assert len(raw_papers) == 11
    assert batch_calls[failed_batch] == len(arxiv_retriever.ARXIV_RETRY_BACKOFF_SECONDS) + 1
    assert sleep_values == [
        arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS,
        arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS,
        *arxiv_retriever.ARXIV_RETRY_BACKOFF_SECONDS,
        arxiv_retriever.ARXIV_REQUEST_INTERVAL_SECONDS,
    ]
    assert any("arXiv API HTTP 503" in warning for warning in warnings)
    assert any("Falling back to RSS data for arXiv batch 2/3" in warning for warning in warnings)
    assert raw_papers[5].summary == "Abstract 5"


def test_arxiv_retriever_limits_feed_entries_from_output_count(config, monkeypatch):
    config.executor.max_paper_num = 5
    feed = _make_arxiv_feed(30)
    requested_batches: list[list[str]] = []

    class FakeSearch:
        def __init__(self, id_list):
            self.id_list = list(id_list)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            requested_batches.append(search.id_list)
            return iter([SimpleNamespace(title=paper_id) for paper_id in search.id_list])

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Search", FakeSearch)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    raw_papers = ArxivRetriever(config)._retrieve_raw_papers()

    assert len(raw_papers) == 20
    assert [len(batch) for batch in requested_batches] == [5, 5, 5, 5]


def test_arxiv_retriever_filters_feed_entries_by_keywords(config, monkeypatch):
    config.source.arxiv.include_keywords = ["few-shot", "sonar"]
    config.source.arxiv.exclude_keywords = ["scaling law"]
    feed = feedparser.FeedParserDict(
        {
            "feed": feedparser.FeedParserDict({"title": "cs.LG updates on arXiv.org"}),
            "entries": [
                feedparser.FeedParserDict(
                    {
                        "id": "oai:arXiv.org:2601.00001v1",
                        "arxiv_announce_type": "new",
                        "title": "Few-Shot Open-Set Recognition for Acoustic Targets",
                        "summary": "Abstract: A transferable prototype method for target recognition.",
                        "dc_creator": "Author 1",
                    }
                ),
                feedparser.FeedParserDict(
                    {
                        "id": "oai:arXiv.org:2601.00002v1",
                        "arxiv_announce_type": "new",
                        "title": "Scaling Law for Large Language Models",
                        "summary": "Abstract: A data scaling law for pretraining.",
                        "dc_creator": "Author 2",
                    }
                ),
                feedparser.FeedParserDict(
                    {
                        "id": "oai:arXiv.org:2601.00003v1",
                        "arxiv_announce_type": "new",
                        "title": "Sonar Domain Adaptation with Unknown Classes",
                        "summary": "Abstract: A method for passive sonar classification.",
                        "dc_creator": "Author 3",
                    }
                ),
            ],
        }
    )
    requested_batches: list[list[str]] = []

    class FakeSearch:
        def __init__(self, id_list):
            self.id_list = list(id_list)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            requested_batches.append(search.id_list)
            return iter([SimpleNamespace(title=paper_id) for paper_id in search.id_list])

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Search", FakeSearch)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    raw_papers = ArxivRetriever(config)._retrieve_raw_papers()

    assert len(raw_papers) == 2
    assert requested_batches == [["2601.00001v1", "2601.00003v1"]]


def test_arxiv_retriever_handles_feed_without_title(config, monkeypatch):
    feed = feedparser.FeedParserDict(
        {
            "feed": feedparser.FeedParserDict(),
            "entries": [],
        }
    )

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    assert ArxivRetriever(config)._retrieve_raw_papers() == []


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=5, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=5, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
