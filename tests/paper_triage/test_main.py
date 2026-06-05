from pathlib import Path
from types import SimpleNamespace

import paper_triage.main as main_module
from paper_triage.arxiv_email_parser import EmailPaper
from paper_triage.paper_analyzer import PaperAnalysis


class FakeReader:
    def fetch_latest_message(self, query):
        return SimpleNamespace(
            subject="Daily arXiv",
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
    report_path = tmp_path / "2026-06-05_paper_triage_report.docx"

    monkeypatch.setattr(main_module, "_load_config", lambda path: {
        "gmail": {"query": 'subject:"Daily arXiv"'},
        "papers": {"max_papers": 5},
        "report": {"output_dir": str(tmp_path)},
        "email_report": {
            "enabled": True,
            "sender": "sender@example.com",
            "receiver": "receiver@example.com",
            "sender_password": "secret",
        },
    })
    monkeypatch.setattr(main_module, "GmailReader", SimpleNamespace(from_env=lambda: FakeReader()))
    monkeypatch.setattr(main_module, "parse_arxiv_email", lambda body, max_papers: [
        EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1")
    ])
    monkeypatch.setattr(main_module, "PaperAnalyzer", SimpleNamespace(from_config=lambda config: FakeAnalyzer()))

    def fake_write_report(papers, analyses, output_dir, diagnostics=None, llm_diagnostics=None):
        report_path.write_bytes(b"fake docx")
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

