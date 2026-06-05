from __future__ import annotations

import mimetypes
import smtplib
from collections import Counter
from datetime import date
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Any

from loguru import logger

from .paper_analyzer import PaperAnalysis


def is_email_enabled(email_config: dict[str, Any] | None) -> bool:
    if not email_config:
        return False
    return _as_bool(email_config.get("enabled", False))


def send_report_email(
    email_config: dict[str, Any],
    report_path: str | Path,
    analyses: list[PaperAnalysis],
    report_date: date | None = None,
) -> None:
    report_path = Path(report_path)
    report_date = report_date or date.today()
    sender = str(email_config.get("sender") or "")
    receiver = str(email_config.get("receiver") or "")
    password = str(email_config.get("sender_password") or "")
    smtp_server = str(email_config.get("smtp_server") or "smtp.qq.com")
    smtp_port = int(email_config.get("smtp_port") or 465)

    missing = [
        name
        for name, value in {
            "SENDER": sender,
            "RECEIVER": receiver,
            "SENDER_PASSWORD": password,
        }.items()
        if not value or value == "None"
    ]
    if missing:
        raise ValueError(f"Cannot send paper triage report email. Missing secrets: {', '.join(missing)}")
    if not report_path.exists():
        raise FileNotFoundError(f"Report attachment does not exist: {report_path}")

    message = EmailMessage()
    message["From"] = formataddr(("Github Action", sender))
    message["To"] = receiver
    message["Subject"] = f"每日 arXiv 论文迁移可行性分析 {report_date:%Y-%m-%d}"
    message.set_content(_build_email_body(analyses, report_path, report_date))

    content_type, _ = mimetypes.guess_type(report_path.name)
    maintype, subtype = (content_type or "application/octet-stream").split("/", 1)
    message.add_attachment(
        report_path.read_bytes(),
        maintype=maintype,
        subtype=subtype,
        filename=report_path.name,
    )

    logger.info("Sending paper triage report email to {}", receiver)
    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(message)
    else:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
    logger.info("Paper triage report email sent")


def _build_email_body(analyses: list[PaperAnalysis], report_path: Path, report_date: date) -> str:
    counts = Counter(analysis.reading_recommendation for analysis in analyses)
    failures = sum(1 for analysis in analyses if analysis.error)
    return (
        f"每日 arXiv 论文迁移可行性分析报告已生成。\n\n"
        f"报告日期：{report_date:%Y-%m-%d}\n"
        f"附件文件：{report_path.name}\n"
        f"阅读建议统计：精读 {counts.get('精读', 0)} 篇，"
        f"略读 {counts.get('略读', 0)} 篇，跳过 {counts.get('跳过', 0)} 篇。\n"
        f"LLM 分析失败：{failures} 篇。\n\n"
        "完整结果见附件 Word 文档。GitHub Actions 页面也保留了 artifact 备份。"
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

