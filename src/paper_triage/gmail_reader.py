from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from loguru import logger


GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


@dataclass(frozen=True)
class GmailMessage:
    message_id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    html: str
    plain: str


class GmailReader:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str) -> None:
        if not client_id or not client_secret or not refresh_token:
            raise ValueError(
                "Gmail OAuth is not configured. Set GMAIL_CLIENT_ID, "
                "GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN in GitHub Secrets."
            )
        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=[GMAIL_READONLY_SCOPE],
        )
        credentials.refresh(Request())
        self._service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    @classmethod
    def from_env(cls) -> "GmailReader":
        return cls(
            client_id=os.getenv("GMAIL_CLIENT_ID", ""),
            client_secret=os.getenv("GMAIL_CLIENT_SECRET", ""),
            refresh_token=os.getenv("GMAIL_REFRESH_TOKEN", ""),
        )

    def fetch_latest_message(self, query: str) -> GmailMessage:
        return self.wait_for_message(
            query=query,
            expected_subject=None,
            attempts=1,
            interval_seconds=0,
            max_results=1,
        )

    def wait_for_message(
        self,
        query: str,
        expected_subject: str | None,
        attempts: int = 12,
        interval_seconds: int = 50,
        max_results: int = 10,
    ) -> GmailMessage:
        attempts = max(1, attempts)
        max_results = max(1, max_results)
        expected = expected_subject.strip() if expected_subject else None
        logger.info(
            "Searching Gmail with query: {}; expected subject: {}",
            query,
            expected or "(latest message)",
        )

        for attempt in range(1, attempts + 1):
            for message in self._fetch_candidate_messages(query, max_results=max_results):
                if expected is None or message.subject.strip() == expected:
                    logger.info("Loaded Gmail message {} with subject {}", message.message_id, message.subject)
                    return message

            if attempt < attempts:
                logger.info(
                    "Expected Gmail message not available yet (attempt {}/{}); waiting {} seconds",
                    attempt,
                    attempts,
                    interval_seconds,
                )
                time.sleep(interval_seconds)

        subject_detail = f" and subject: {expected}" if expected else ""
        raise RuntimeError(f"No Gmail message matched query: {query}{subject_detail} after {attempts} attempts")

    def _fetch_candidate_messages(self, query: str, max_results: int) -> list[GmailMessage]:
        listing = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        messages = listing.get("messages", [])
        candidates = []
        for message in messages:
            message_id = message["id"]
            raw_message = (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
            candidates.append(_to_gmail_message(raw_message))
        return candidates


def _to_gmail_message(raw_message: dict[str, Any]) -> GmailMessage:
    payload = raw_message.get("payload", {})
    headers = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}
    html, plain = _extract_bodies(payload)
    date_header = headers.get("date", "")
    try:
        date = parsedate_to_datetime(date_header).isoformat() if date_header else ""
    except (TypeError, ValueError):
        date = date_header

    return GmailMessage(
        message_id=raw_message.get("id", ""),
        thread_id=raw_message.get("threadId", ""),
        subject=headers.get("subject", ""),
        sender=headers.get("from", ""),
        date=date,
        html=html,
        plain=plain,
    )


def _extract_bodies(payload: dict[str, Any]) -> tuple[str, str]:
    html_parts: list[str] = []
    plain_parts: list[str] = []

    def visit(part: dict[str, Any]) -> None:
        mime_type = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if body_data:
            decoded = _decode_body(body_data)
            if mime_type == "text/html":
                html_parts.append(decoded)
            elif mime_type == "text/plain":
                plain_parts.append(decoded)

        for child in part.get("parts", []) or []:
            visit(child)

    visit(payload)
    return "\n".join(html_parts), "\n".join(plain_parts)


def _decode_body(data: str) -> str:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="replace")

