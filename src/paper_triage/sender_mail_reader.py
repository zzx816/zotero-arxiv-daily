from __future__ import annotations

import email
import imaplib
import os
import re
from datetime import date, datetime, timedelta
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parsedate_to_datetime
from typing import Any

from loguru import logger

from .gmail_reader import GmailMessage


class SenderMailboxReader:
    def __init__(
        self,
        username: str,
        password: str,
        imap_server: str,
        imap_port: int = 993,
        mailbox_hints: list[str] | None = None,
    ) -> None:
        if not username or not password:
            raise ValueError("Sender mailbox fallback is not configured. Missing sender email or sender password.")
        if not imap_server:
            raise ValueError("Sender mailbox fallback is not configured. Missing IMAP server.")
        self.username = username
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.mailbox_hints = mailbox_hints or []

    @classmethod
    def from_config(cls, email_config: dict[str, Any]) -> "SenderMailboxReader":
        sender = str(email_config.get("sender") or os.getenv("SENDER") or "")
        password = str(email_config.get("sender_password") or os.getenv("SENDER_PASSWORD") or "")
        smtp_server = str(email_config.get("smtp_server") or os.getenv("SMTP_SERVER") or "smtp.qq.com")
        imap_server = os.getenv("IMAP_SERVER") or _imap_server_from_smtp(smtp_server)
        imap_port = int(os.getenv("IMAP_PORT") or 993)
        mailbox_hints = [item.strip() for item in os.getenv("IMAP_MAILBOX_HINTS", "").split(",") if item.strip()]
        return cls(
            username=sender,
            password=password,
            imap_server=imap_server,
            imap_port=imap_port,
            mailbox_hints=mailbox_hints,
        )

    def find_recent_daily_message(
        self,
        expected_subject: str,
        report_date: date,
        max_age_hours: int = 36,
        max_messages: int = 30,
    ) -> GmailMessage:
        logger.info("Falling back to sender mailbox IMAP {} for {}", self.imap_server, self.username)
        with imaplib.IMAP4_SSL(self.imap_server, self.imap_port) as client:
            client.login(self.username, self.password)
            mailbox_names = _candidate_mailboxes(_list_mailboxes(client), self.mailbox_hints)
            best_recent: GmailMessage | None = None

            for mailbox in mailbox_names:
                for message in _iter_recent_messages(client, mailbox, max_messages=max_messages):
                    if message.subject.strip() == expected_subject:
                        logger.info("Loaded sender mailbox message {} from {}", message.subject, mailbox)
                        return message
                    if best_recent is None and _is_recent_daily_message(
                        message.subject,
                        message.date,
                        report_date,
                        max_age_hours=max_age_hours,
                    ):
                        best_recent = message

            if best_recent is not None:
                logger.info("Loaded recent sender mailbox fallback message {}", best_recent.subject)
                return best_recent

        raise RuntimeError(
            f"Sender mailbox fallback did not find a recent Daily arXiv message matching {expected_subject!r}"
        )


def _iter_recent_messages(client: imaplib.IMAP4_SSL, mailbox: str, max_messages: int):
    try:
        status, _ = client.select(_quote_mailbox(mailbox), readonly=True)
    except imaplib.IMAP4.error:
        return
    if status != "OK":
        return

    status, data = client.search(None, "ALL")
    if status != "OK" or not data or not data[0]:
        return

    message_ids = data[0].split()
    for message_id in reversed(message_ids[-max(1, max_messages) :]):
        status, fetch_data = client.fetch(message_id, "(RFC822)")
        if status != "OK":
            continue
        raw_bytes = _extract_rfc822_bytes(fetch_data)
        if raw_bytes is None:
            continue
        yield _to_message(raw_bytes, f"{mailbox}:{message_id.decode(errors='ignore')}")


def _list_mailboxes(client: imaplib.IMAP4_SSL) -> list[str]:
    status, data = client.list()
    if status != "OK" or not data:
        return []
    mailbox_names = []
    for item in data:
        if not item:
            continue
        mailbox = _parse_mailbox_name(item)
        if mailbox:
            mailbox_names.append(mailbox)
    return mailbox_names


def _candidate_mailboxes(mailboxes: list[str], mailbox_hints: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    preferred_terms = mailbox_hints or [
        "sent",
        "sent messages",
        "sent mail",
        "inbox",
    ]

    def add(name: str) -> None:
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    lower_map = {name.lower(): name for name in mailboxes}
    for term in preferred_terms:
        term = term.lower()
        for lower_name, original_name in lower_map.items():
            if term in lower_name:
                add(original_name)

    for name in mailboxes:
        add(name)
    return ordered


def _parse_mailbox_name(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    match = re.search(r'"((?:[^"\\]|\\.)*)"\s*$', text)
    if match:
        return match.group(1).replace('\\"', '"')
    parts = text.split(" ")
    return parts[-1].strip('"') if parts else ""


def _quote_mailbox(mailbox: str) -> str:
    return f'"{mailbox.replace(chr(34), chr(92) + chr(34))}"'


def _extract_rfc822_bytes(fetch_data: list[Any]) -> bytes | None:
    for item in fetch_data:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
            return bytes(item[1])
    return None


def _to_message(raw_bytes: bytes, message_id: str) -> GmailMessage:
    parsed = email.message_from_bytes(raw_bytes)
    subject = _decode_mime_header(parsed.get("Subject", ""))
    sender = _decode_mime_header(parsed.get("From", ""))
    date_header = parsed.get("Date", "")
    try:
        date_value = parsedate_to_datetime(date_header).isoformat() if date_header else ""
    except (TypeError, ValueError):
        date_value = date_header
    html, plain = _extract_bodies(parsed)
    return GmailMessage(
        message_id=message_id,
        thread_id="",
        subject=subject,
        sender=sender,
        date=date_value,
        html=html,
        plain=plain,
    )


def _extract_bodies(message: Message) -> tuple[str, str]:
    html_parts: list[str] = []
    plain_parts: list[str] = []

    for part in message.walk():
        if part.is_multipart():
            continue
        content_type = part.get_content_type()
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="replace")
        except LookupError:
            decoded = payload.decode("utf-8", errors="replace")
        if content_type == "text/html":
            html_parts.append(decoded)
        elif content_type == "text/plain":
            plain_parts.append(decoded)

    return "\n".join(html_parts), "\n".join(plain_parts)


def _decode_mime_header(value: str) -> str:
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _parse_subject_date(subject: str) -> date | None:
    prefix = "Daily arXiv "
    if not subject.startswith(prefix):
        return None
    try:
        return datetime.strptime(subject[len(prefix) :].strip(), "%Y/%m/%d").date()
    except ValueError:
        return None


def _is_recent_daily_message(subject: str, message_date: str, report_date: date, max_age_hours: int) -> bool:
    subject_date = _parse_subject_date(subject)
    if subject_date is not None and subject_date not in {report_date, report_date - timedelta(days=1)}:
        return False

    if not message_date:
        return subject.startswith("Daily arXiv ")

    try:
        timestamp = datetime.fromisoformat(message_date)
    except ValueError:
        return subject.startswith("Daily arXiv ")

    age = datetime.now(timestamp.tzinfo) - timestamp
    if age < timedelta(0):
        age = timedelta(0)
    return subject.startswith("Daily arXiv ") and age <= timedelta(hours=max_age_hours)


def _imap_server_from_smtp(smtp_server: str) -> str:
    server = (smtp_server or "").strip().lower()
    mapping = {
        "smtp.qq.com": "imap.qq.com",
        "smtp.gmail.com": "imap.gmail.com",
        "smtp-mail.outlook.com": "outlook.office365.com",
        "smtp.office365.com": "outlook.office365.com",
        "smtp.163.com": "imap.163.com",
        "smtp.126.com": "imap.126.com",
    }
    if server in mapping:
        return mapping[server]
    if server.startswith("smtp."):
        return "imap." + server[len("smtp.") :]
    return server
