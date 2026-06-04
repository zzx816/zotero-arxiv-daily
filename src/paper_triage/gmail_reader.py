from __future__ import annotations

import base64
import os
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
        logger.info("Searching Gmail with query: {}", query)
        listing = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=1)
            .execute()
        )
        messages = listing.get("messages", [])
        if not messages:
            raise RuntimeError(f"No Gmail message matched query: {query}")

        message_id = messages[0]["id"]
        raw_message = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        logger.info("Loaded Gmail message {}", message_id)
        return _to_gmail_message(raw_message)


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

