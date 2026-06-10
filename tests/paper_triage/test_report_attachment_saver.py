import base64

from paper_triage.gmail_reader import GmailReader
from paper_triage.report_attachment_saver import save_report_attachments


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class FakeAttachmentsResource:
    def __init__(self, attachments):
        self.attachments = attachments

    def get(self, userId, messageId, id):
        return FakeRequest(self.attachments[(messageId, id)])


class FakeMessagesResource:
    def __init__(self, raw_messages, attachments):
        self.raw_messages = raw_messages
        self.attachments_resource = FakeAttachmentsResource(attachments)
        self.list_calls = []

    def list(self, userId, q, maxResults):
        self.list_calls.append({"userId": userId, "q": q, "maxResults": maxResults})
        return FakeRequest({"messages": [{"id": message_id} for message_id in self.raw_messages]})

    def get(self, userId, id, format):
        return FakeRequest(self.raw_messages[id])

    def attachments(self):
        return self.attachments_resource


class FakeUsersResource:
    def __init__(self, messages_resource):
        self._messages_resource = messages_resource

    def messages(self):
        return self._messages_resource


class FakeService:
    def __init__(self, raw_messages, attachments):
        self.messages_resource = FakeMessagesResource(raw_messages, attachments)

    def users(self):
        return FakeUsersResource(self.messages_resource)


def test_save_report_attachments_downloads_latest_docx(tmp_path):
    reader = object.__new__(GmailReader)
    reader._service = FakeService(
        raw_messages={
            "message-1": _raw_message(
                "每日 arXiv 论文迁移可行性分析 2026-06-10",
                "2026-06-10_paper_triage_report.docx",
                "attachment-1",
            )
        },
        attachments={
            ("message-1", "attachment-1"): {"data": _encode(b"docx-bytes")},
        },
    )

    saved = save_report_attachments(output_dir=tmp_path, reader=reader)

    assert len(saved) == 1
    assert saved[0].status == "saved"
    assert saved[0].filename == "2026-06-10_paper_triage_report.docx"
    assert saved[0].path.read_bytes() == b"docx-bytes"


def test_save_report_attachments_keeps_existing_file(tmp_path):
    existing = tmp_path / "2026-06-10_paper_triage_report.docx"
    existing.write_bytes(b"existing")
    reader = object.__new__(GmailReader)
    reader._service = FakeService(
        raw_messages={
            "message-1": _raw_message(
                "每日 arXiv 论文迁移可行性分析 2026-06-10",
                existing.name,
                "attachment-1",
            )
        },
        attachments={
            ("message-1", "attachment-1"): {"data": _encode(b"new")},
        },
    )

    saved = save_report_attachments(output_dir=tmp_path, reader=reader)

    assert saved[0].status == "exists"
    assert existing.read_bytes() == b"existing"


def _raw_message(subject, filename, attachment_id):
    return {
        "id": "message-1",
        "payload": {
            "headers": [{"name": "Subject", "value": subject}],
            "parts": [
                {
                    "filename": filename,
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "body": {"attachmentId": attachment_id},
                }
            ],
        },
    }


def _encode(data):
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
