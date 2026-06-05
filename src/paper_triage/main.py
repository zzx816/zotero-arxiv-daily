from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf

from .arxiv_email_parser import parse_arxiv_email
from .email_sender import is_email_enabled, send_report_email
from .gmail_reader import GmailReader
from .paper_analyzer import PaperAnalyzer
from .report_writer import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a daily arXiv paper triage report from Gmail.")
    parser.add_argument("--config", default="config/paper_triage_config.yaml", help="Path to triage config YAML.")
    args = parser.parse_args(argv)

    _configure_logging()
    config = _load_config(args.config)

    try:
        reader = GmailReader.from_env()
        message = reader.fetch_latest_message(config["gmail"]["query"])
    except Exception as exc:
        logger.error("Failed to read Gmail: {}", exc)
        return 1

    email_body = message.html or message.plain
    max_papers = int(config.get("papers", {}).get("max_papers", 5))
    papers = parse_arxiv_email(email_body, max_papers=max_papers)
    diagnostics: list[str] = [
        f"Gmail message: {message.subject or '(no subject)'}",
        f"Message date: {message.date or '(unknown)'}",
    ]

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
