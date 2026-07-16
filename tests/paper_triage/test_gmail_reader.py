import pytest

from paper_triage.gmail_reader import GmailReader


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class FakeMessagesResource:
    def __init__(self, batches, raw_messages):
        self.batches = list(batches)
        self.raw_messages = raw_messages
        self.list_calls = []

    def list(self, userId, q, maxResults, includeSpamTrash):
        self.list_calls.append(
            {
                "userId": userId,
                "q": q,
                "maxResults": maxResults,
                "includeSpamTrash": includeSpamTrash,
            }
        )
        batch = self.batches.pop(0) if self.batches else []
        return FakeRequest({"messages": [{"id": message_id} for message_id in batch]})

    def get(self, userId, id, format):
        return FakeRequest(self.raw_messages[id])


class FakeUsersResource:
    def __init__(self, messages_resource):
        self._messages_resource = messages_resource

    def messages(self):
        return self._messages_resource


class FakeService:
    def __init__(self, batches, raw_messages):
        self.messages_resource = FakeMessagesResource(batches, raw_messages)

    def users(self):
        return FakeUsersResource(self.messages_resource)


def test_wait_for_message_returns_matching_subject_immediately(monkeypatch):
    reader = _reader_with_service(
        batches=[["today"]],
        raw_messages={"today": _raw_message("today", "Daily arXiv 2026/06/08")},
    )
    sleeps = []
    monkeypatch.setattr("paper_triage.gmail_reader.time.sleep", sleeps.append)

    message = reader.wait_for_message(
        query='subject:"Daily arXiv"',
        expected_subject="Daily arXiv 2026/06/08",
        attempts=3,
        interval_seconds=50,
        max_results=10,
    )

    assert message.message_id == "today"
    assert message.subject == "Daily arXiv 2026/06/08"
    assert sleeps == []
    assert reader._service.messages_resource.list_calls == [
        {
            "userId": "me",
            "q": 'subject:"Daily arXiv"',
            "maxResults": 10,
            "includeSpamTrash": True,
        }
    ]


def test_wait_for_message_retries_until_expected_subject(monkeypatch):
    reader = _reader_with_service(
        batches=[["yesterday"], ["today"]],
        raw_messages={
            "yesterday": _raw_message("yesterday", "Daily arXiv 2026/06/07"),
            "today": _raw_message("today", "Daily arXiv 2026/06/08"),
        },
    )
    sleeps = []
    monkeypatch.setattr("paper_triage.gmail_reader.time.sleep", sleeps.append)

    message = reader.wait_for_message(
        query='subject:"Daily arXiv"',
        expected_subject="Daily arXiv 2026/06/08",
        attempts=2,
        interval_seconds=50,
        max_results=5,
    )

    assert message.message_id == "today"
    assert sleeps == [50]
    assert len(reader._service.messages_resource.list_calls) == 2


def test_wait_for_message_raises_after_attempts(monkeypatch):
    reader = _reader_with_service(
        batches=[["yesterday"], ["older"]],
        raw_messages={
            "yesterday": _raw_message("yesterday", "Daily arXiv 2026/06/07"),
            "older": _raw_message("older", "Daily arXiv 2026/06/06"),
        },
    )
    monkeypatch.setattr("paper_triage.gmail_reader.time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="Daily arXiv 2026/06/08"):
        reader.wait_for_message(
            query='subject:"Daily arXiv"',
            expected_subject="Daily arXiv 2026/06/08",
            attempts=2,
            interval_seconds=50,
            max_results=5,
        )


def _reader_with_service(batches, raw_messages):
    reader = object.__new__(GmailReader)
    reader._service = FakeService(batches, raw_messages)
    return reader


def _raw_message(message_id, subject):
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "Github Action <sender@example.com>"},
                {"name": "Date", "value": "Mon, 8 Jun 2026 11:26:37 +0000"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "SGVsbG8="},
        },
    }
