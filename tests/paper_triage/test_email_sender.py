from paper_triage.email_sender import is_email_enabled, send_report_email
from paper_triage.paper_analyzer import PaperAnalysis


class FakeSMTP:
    def __init__(self, server, port):
        self.server = server
        self.port = port
        self.logged_in = None
        self.message = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def login(self, sender, password):
        self.logged_in = (sender, password)

    def send_message(self, message):
        self.message = message


def test_send_report_email_attaches_docx(monkeypatch, tmp_path):
    smtp_instances = []

    def fake_smtp_ssl(server, port):
        instance = FakeSMTP(server, port)
        smtp_instances.append(instance)
        return instance

    monkeypatch.setattr("paper_triage.email_sender.smtplib.SMTP_SSL", fake_smtp_ssl)
    report_path = tmp_path / "2026-06-05_paper_triage_report.docx"
    report_path.write_bytes(b"fake docx")

    send_report_email(
        {
            "enabled": True,
            "sender": "sender@example.com",
            "receiver": "receiver@example.com",
            "sender_password": "secret",
            "smtp_server": "smtp.example.com",
            "smtp_port": 465,
        },
        report_path,
        [
            PaperAnalysis(
                relevance_score=8,
                matched_research_direction="开放集识别 OSR",
                core_contribution="贡献",
                method_type="prototype learning",
                reading_recommendation="精读",
                keyword_tags=["OSR"],
            )
        ],
    )

    smtp = smtp_instances[0]
    assert smtp.server == "smtp.example.com"
    assert smtp.port == 465
    assert smtp.logged_in == ("sender@example.com", "secret")
    assert smtp.message["To"] == "receiver@example.com"
    assert "每日 arXiv 论文迁移可行性分析" in smtp.message["Subject"]
    attachments = list(smtp.message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == report_path.name


def test_is_email_enabled_accepts_string_true():
    assert is_email_enabled({"enabled": "true"}) is True
    assert is_email_enabled({"enabled": "false"}) is False

