from __future__ import annotations

import argparse
import base64
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
from loguru import logger

from .gmail_reader import GmailReader


DEFAULT_QUERY = 'subject:"每日 arXiv 论文迁移可行性分析" filename:docx newer_than:14d'
DEFAULT_OUTPUT_DIR = r"D:\Downloads\lunwen"


@dataclass(frozen=True)
class SavedAttachment:
    message_id: str
    subject: str
    filename: str
    path: Path
    status: str


def save_report_attachments(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    query: str = DEFAULT_QUERY,
    max_results: int = 10,
    overwrite: bool = False,
    reader: GmailReader | None = None,
) -> list[SavedAttachment]:
    reader = reader or GmailReader.from_env()
    service = reader._service
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    listing = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max(1, max_results))
        .execute()
    )
    messages = listing.get("messages", [])
    saved: list[SavedAttachment] = []

    for message_ref in messages:
        message_id = message_ref["id"]
        raw_message = (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        subject = _header(raw_message, "subject")
        for attachment in _docx_attachments(raw_message.get("payload", {})):
            filename = _safe_filename(attachment["filename"])
            target = output_path / filename
            if target.exists() and not overwrite:
                saved.append(SavedAttachment(message_id, subject, filename, target, "exists"))
                continue

            data = attachment.get("data")
            if data is None:
                attachment_id = attachment["attachment_id"]
                attachment_body = (
                    service.users()
                    .messages()
                    .attachments()
                    .get(userId="me", messageId=message_id, id=attachment_id)
                    .execute()
                )
                data = attachment_body.get("data", "")

            target.write_bytes(_decode_base64url(data))
            saved.append(SavedAttachment(message_id, subject, filename, target, "saved"))

        if saved:
            break

    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save the latest paper triage Word report from Gmail.")
    parser.add_argument("--output-dir", default=os.getenv("PAPER_TRIAGE_LOCAL_REPORT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--query", default=os.getenv("PAPER_TRIAGE_REPORT_QUERY", DEFAULT_QUERY))
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    dotenv.load_dotenv()
    _configure_logging()

    try:
        saved = save_report_attachments(
            output_dir=args.output_dir,
            query=args.query,
            max_results=args.max_results,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        logger.error("Failed to save report attachment: {}", exc)
        return 1

    if not saved:
        logger.warning("No report .docx attachment matched query: {}", args.query)
        return 0

    for item in saved:
        logger.info("{} {} -> {}", item.status, item.filename, item.path)
    return 0


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def _header(raw_message: dict[str, Any], name: str) -> str:
    headers = raw_message.get("payload", {}).get("headers", [])
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return str(header.get("value", ""))
    return ""


def _docx_attachments(payload: dict[str, Any]) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []

    def visit(part: dict[str, Any]) -> None:
        filename = str(part.get("filename") or "")
        body = part.get("body", {}) or {}
        if filename.lower().endswith(".docx"):
            attachment: dict[str, str] = {"filename": filename}
            if body.get("attachmentId"):
                attachment["attachment_id"] = str(body["attachmentId"])
            if body.get("data"):
                attachment["data"] = str(body["data"])
            if "attachment_id" in attachment or "data" in attachment:
                attachments.append(attachment)

        for child in part.get("parts", []) or []:
            visit(child)

    visit(payload)
    return attachments


def _decode_base64url(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _safe_filename(filename: str) -> str:
    cleaned = "".join("_" if char in '<>:"/\\|?*' else char for char in filename).strip()
    return cleaned or "paper_triage_report.docx"


if __name__ == "__main__":
    raise SystemExit(main())
