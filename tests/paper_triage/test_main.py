from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import paper_triage.main as main_module
from paper_triage.arxiv_email_parser import EmailPaper
from paper_triage.paper_analyzer import PaperAnalysis


class FakeReader:
    def wait_for_message(self, query, expected_subject, attempts, interval_seconds, max_results):
        self.query = query
        self.expected_subject = expected_subject
        self.attempts = attempts
        self.interval_seconds = interval_seconds
        self.max_results = max_results
        return SimpleNamespace(
            subject=expected_subject,
            date="2026-06-05",
            html="<html></html>",
            plain="",
        )


class FakeAnalyzer:
    config_summary = {
        "provider": "SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "key_source": "SILICONFLOW_API_KEY",
        "warnings": [],
    }

    def analyze(self, papers):
        return [
            PaperAnalysis(
                relevance_score=8,
                matched_research_direction="开放集识别 OSR",
                core_contribution="贡献",
                method_type="prototype learning",
                reading_recommendation="精读",
                keyword_tags=["OSR"],
            )
        ]


def test_main_sends_report_email_when_enabled(monkeypatch, tmp_path):
    sent = {}
    fake_reader = FakeReader()
    report_path = tmp_path / "2026-06-05_paper_triage_report.docx"

    monkeypatch.setattr(main_module, "_load_config", lambda path: {
        "gmail": {
            "query": 'subject:"Daily arXiv"',
            "wait_attempts": 2,
            "wait_interval_seconds": 3,
            "max_results": 4,
        },
        "papers": {"max_papers": 5},
        "report": {"output_dir": str(tmp_path)},
        "email_report": {
            "enabled": True,
            "sender": "sender@example.com",
            "receiver": "receiver@example.com",
            "sender_password": "secret",
        },
    })
    monkeypatch.setattr(main_module, "GmailReader", SimpleNamespace(from_env=lambda: fake_reader))
    monkeypatch.setattr(main_module, "parse_arxiv_email", lambda body, max_papers: [
        EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1")
    ])
    monkeypatch.setattr(main_module, "PaperAnalyzer", SimpleNamespace(from_config=lambda config: FakeAnalyzer()))

    def fake_write_report(papers, analyses, output_dir, diagnostics=None, llm_diagnostics=None):
        report_path.write_bytes(b"fake docx")
        sent["diagnostics"] = diagnostics
        sent["llm_diagnostics"] = llm_diagnostics
        return report_path

    def fake_send_report_email(email_config, path, analyses):
        sent["email_config"] = email_config
        sent["path"] = Path(path)
        sent["analyses"] = analyses

    monkeypatch.setattr(main_module, "write_report", fake_write_report)
    monkeypatch.setattr(main_module, "send_report_email", fake_send_report_email)

    assert main_module.main(["--config", "ignored.yaml"]) == 0
    assert sent["path"] == report_path
    assert sent["email_config"]["receiver"] == "receiver@example.com"
    assert sent["llm_diagnostics"]["provider"] == "SiliconFlow"
    assert fake_reader.query == 'subject:"Daily arXiv"'
    assert fake_reader.expected_subject.startswith("Daily arXiv ")
    assert fake_reader.attempts == 2
    assert fake_reader.interval_seconds == 3
    assert fake_reader.max_results == 4
    assert f"Expected Gmail subject: {fake_reader.expected_subject}" in sent["diagnostics"]


def test_main_falls_back_to_latest_recent_daily_email(monkeypatch, tmp_path):
    sent = {}
    expected_subject = "Daily arXiv 2026/06/12"

    class FallbackReader:
        def wait_for_message(self, query, expected_subject, attempts, interval_seconds, max_results):
            raise RuntimeError("exact subject not found in time")

        def fetch_latest_message(self, query):
            return SimpleNamespace(
                subject=expected_subject,
                date=(datetime.now().astimezone() - timedelta(minutes=5)).isoformat(),
                html="<html></html>",
                plain="",
            )

    report_path = tmp_path / "2026-06-12_paper_triage_report.docx"

    monkeypatch.setattr(main_module, "_load_config", lambda path: {
        "gmail": {
            "query": 'subject:"Daily arXiv"',
            "wait_attempts": 2,
            "wait_interval_seconds": 3,
            "max_results": 4,
            "fallback_max_age_hours": 36,
        },
        "papers": {"max_papers": 5},
        "report": {"output_dir": str(tmp_path)},
        "email_report": {"enabled": False},
    })
    monkeypatch.setattr(main_module, "GmailReader", SimpleNamespace(from_env=lambda: FallbackReader()))
    monkeypatch.setattr(main_module, "parse_arxiv_email", lambda body, max_papers: [
        EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1")
    ])
    monkeypatch.setattr(main_module, "PaperAnalyzer", SimpleNamespace(from_config=lambda config: FakeAnalyzer()))

    def fake_write_report(papers, analyses, output_dir, diagnostics=None, llm_diagnostics=None):
        report_path.write_bytes(b"fake docx")
        sent["diagnostics"] = diagnostics
        return report_path

    monkeypatch.setattr(main_module, "write_report", fake_write_report)

    assert main_module.main(["--config", "ignored.yaml"]) == 0
    assert "Used fallback Gmail lookup after exact-subject wait timed out." in sent["diagnostics"]


def test_recent_daily_message_rejects_stale_subject_date():
    assert not main_module._is_recent_daily_message(
        "Daily arXiv 2026/06/10",
        datetime.now().astimezone().isoformat(),
        report_date=datetime(2026, 6, 12).date(),
        max_age_hours=36,
    )


def test_recent_daily_message_accepts_recent_previous_day_subject():
    assert main_module._is_recent_daily_message(
        "Daily arXiv 2026/06/11",
        datetime.now().astimezone().isoformat(),
        report_date=datetime(2026, 6, 12).date(),
        max_age_hours=36,
    )


def test_main_falls_back_to_sender_mailbox_when_gmail_token_is_revoked(monkeypatch, tmp_path):
    sent = {}
    expected_subject = "Daily arXiv 2026/06/12"

    class FakeSenderReader:
        def find_recent_daily_message(self, expected_subject, report_date, max_age_hours, max_messages):
            return SimpleNamespace(
                subject=expected_subject,
                date=(datetime.now().astimezone() - timedelta(minutes=2)).isoformat(),
                html="<html></html>",
                plain="",
            )

    report_path = tmp_path / "2026-06-12_paper_triage_report.docx"

    monkeypatch.setattr(main_module, "_load_config", lambda path: {
        "gmail": {
            "query": 'subject:"Daily arXiv"',
            "wait_attempts": 2,
            "wait_interval_seconds": 3,
            "max_results": 4,
            "fallback_max_age_hours": 36,
        },
        "papers": {"max_papers": 5},
        "report": {"output_dir": str(tmp_path)},
        "email_report": {
            "enabled": False,
            "sender": "sender@qq.com",
            "sender_password": "secret",
            "smtp_server": "smtp.qq.com",
        },
    })
    monkeypatch.setattr(
        main_module,
        "GmailReader",
        SimpleNamespace(from_env=lambda: (_ for _ in ()).throw(RuntimeError("invalid_grant: token revoked"))),
    )
    monkeypatch.setattr(
        main_module,
        "SenderMailboxReader",
        SimpleNamespace(from_config=lambda config: FakeSenderReader()),
    )
    monkeypatch.setattr(main_module, "parse_arxiv_email", lambda body, max_papers: [
        EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1")
    ])
    monkeypatch.setattr(main_module, "PaperAnalyzer", SimpleNamespace(from_config=lambda config: FakeAnalyzer()))

    def fake_write_report(papers, analyses, output_dir, diagnostics=None, llm_diagnostics=None):
        report_path.write_bytes(b"fake docx")
        sent["diagnostics"] = diagnostics
        return report_path

    monkeypatch.setattr(main_module, "write_report", fake_write_report)

    assert main_module.main(["--config", "ignored.yaml"]) == 0
    assert "Message source: sender-imap-fallback" in sent["diagnostics"]

