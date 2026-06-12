from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf

from .arxiv_email_parser import parse_arxiv_email
from .email_sender import is_email_enabled, send_report_email
from .gmail_reader import GmailReader
from .paper_analyzer import PaperAnalyzer
from .report_writer import write_report
from .sender_mail_reader import SenderMailboxReader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a daily arXiv paper triage report from Gmail.")
    parser.add_argument("--config", default="config/paper_triage_config.yaml", help="Path to triage config YAML.")
    args = parser.parse_args(argv)

    _configure_logging()
    config = _load_config(args.config)

    try:
        gmail_config = config["gmail"]
        email_config = config.get("email_report", {})
        expected_subject = _daily_arxiv_subject(date.today())
        message, used_subject_fallback, message_source = _load_daily_message(
            gmail_config,
            email_config,
            expected_subject,
        )
    except Exception as exc:
        logger.error("Failed to read daily email source: {}", exc)
        return 1

    email_body = message.html or message.plain
    max_papers = int(config.get("papers", {}).get("max_papers", 5))
    papers = parse_arxiv_email(email_body, max_papers=max_papers)
    diagnostics: list[str] = [
        f"Expected Gmail subject: {expected_subject}",
        f"Gmail message: {message.subject or '(no subject)'}",
        f"Message date: {message.date or '(unknown)'}",
        f"Message source: {message_source}",
    ]
    if used_subject_fallback:
        diagnostics.append("Used fallback Gmail lookup after exact-subject wait timed out.")

    if not papers:
        diagnostics.append("未从邮件中解析到论文；请检查邮件 HTML 格式或 Gmail 查询条件。")
        output_path = write_report([], [], config["report"]["output_dir"], diagnostics=diagnostics)
        logger.warning("No papers parsed. Diagnostic report generated at {}", output_path)
        return _send_report_if_enabled(config, output_path, [])

    analyzer = PaperAnalyzer.from_config(config)
    logger.info("LLM config: {}", analyzer.config_summary)
    for warning in analyzer.config_summary.get("warnings", []):
        logger.warning(warning)
    analyses = analyzer.analyze(papers)
    output_path = write_report(
        papers,
        analyses,
        config["report"]["output_dir"],
        diagnostics=diagnostics,
        llm_diagnostics=analyzer.config_summary,
    )
    logger.info("Paper triage completed: {}", output_path)
    return _send_report_if_enabled(config, output_path, analyses)


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(path)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("paper triage config must resolve to a mapping")
    return resolved


def _load_daily_message(gmail_config: dict[str, Any], email_config: dict[str, Any], expected_subject: str):
    try:
        reader = GmailReader.from_env()
        message, used_subject_fallback = _load_daily_message_from_gmail(reader, gmail_config, expected_subject)
        return message, used_subject_fallback, "gmail"
    except Exception as exc:
        if not _should_try_sender_mail_fallback(exc):
            raise
        logger.warning("Gmail reader unavailable ({}); trying sender mailbox IMAP fallback", exc)
        message = _load_daily_message_from_sender_mailbox(email_config, gmail_config, expected_subject)
        return message, False, "sender-imap-fallback"


def _load_daily_message_from_gmail(reader: GmailReader, gmail_config: dict[str, Any], expected_subject: str):
    query = str(gmail_config["query"])
    attempts = int(gmail_config.get("wait_attempts", 30))
    interval_seconds = int(gmail_config.get("wait_interval_seconds", 60))
    max_results = int(gmail_config.get("max_results", 10))
    fallback_max_age_hours = int(gmail_config.get("fallback_max_age_hours", 36))

    try:
        return (
            reader.wait_for_message(
                query=query,
                expected_subject=expected_subject,
                attempts=attempts,
                interval_seconds=interval_seconds,
                max_results=max_results,
            ),
            False,
        )
    except RuntimeError as exc:
        logger.warning(
            "Exact Gmail subject {} not found after {} attempts; trying latest matching daily email fallback",
            expected_subject,
            attempts,
        )
        latest = reader.fetch_latest_message(query)
        if not _is_recent_daily_message(latest.subject, latest.date, date.today(), fallback_max_age_hours):
            raise RuntimeError(
                f"{exc}; latest matching message was subject={latest.subject!r}, date={latest.date!r}"
            ) from exc
        return latest, True


def _load_daily_message_from_sender_mailbox(
    email_config: dict[str, Any],
    gmail_config: dict[str, Any],
    expected_subject: str,
):
    fallback_max_age_hours = int(gmail_config.get("fallback_max_age_hours", 36))
    max_results = int(gmail_config.get("max_results", 10))
    reader = SenderMailboxReader.from_config(email_config)
    return reader.find_recent_daily_message(
        expected_subject=expected_subject,
        report_date=date.today(),
        max_age_hours=fallback_max_age_hours,
        max_messages=max_results * 3,
    )


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


def _should_try_sender_mail_fallback(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    fallback_markers = [
        "invalid_grant",
        "token has been expired or revoked",
        "token has been expired",
        "revoked",
        "gmail oauth is not configured",
        "refresh token",
        "unauthorized_client",
    ]
    return any(marker in text for marker in fallback_markers)


def _parse_subject_date(subject: str) -> date | None:
    prefix = "Daily arXiv "
    if not subject.startswith(prefix):
        return None
    try:
        return datetime.strptime(subject[len(prefix) :].strip(), "%Y/%m/%d").date()
    except ValueError:
        return None


def _daily_arxiv_subject(report_date: date) -> str:
    return f"Daily arXiv {report_date:%Y/%m/%d}"


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def _send_report_if_enabled(config: dict[str, Any], output_path: Path, analyses: list) -> int:
    email_config = config.get("email_report", {})
    if not is_email_enabled(email_config):
        logger.info("Email report sending is disabled")
        return 0
    try:
        send_report_email(email_config, output_path, analyses)
        return 0
    except Exception as exc:
        logger.error("Failed to send report email: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
