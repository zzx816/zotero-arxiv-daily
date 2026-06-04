from datetime import date

from docx import Document

from paper_triage.arxiv_email_parser import EmailPaper
from paper_triage.paper_analyzer import PaperAnalysis
from paper_triage.report_writer import write_report


def test_write_report_creates_docx_with_required_sections(tmp_path):
    paper = EmailPaper(
        title="Open-set prototype learning for acoustic classification",
        abstract="A short abstract.",
        arxiv_url="https://arxiv.org/abs/2401.00001",
    )
    analysis = PaperAnalysis(
        relevance_score=8,
        matched_research_direction="开放集识别 OSR",
        core_contribution="提出开放集原型学习方法。",
        method_type="prototype learning",
        technique_flags={"open-set": True, "prototype learning": True},
        transferable_to_underwater=True,
        transfer_feasibility="高",
        transferable_parts="原型分类和拒识阈值。",
        hard_to_transfer_parts="依赖图像增强的部分。",
        minimal_experiment="在 DeepShip 上提取 Log-Mel 后测试。",
        reading_recommendation="精读",
        keyword_tags=["OSR", "prototype", "DeepShip"],
    )

    path = write_report([paper], [analysis], tmp_path, report_date=date(2026, 6, 4))

    assert path.exists()
    document = Document(path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "每日 arXiv 论文迁移可行性分析报告" in text
    assert "今日总览" in text
    assert "逐篇论文详细分析" in text
    assert "今日精读清单" in text
