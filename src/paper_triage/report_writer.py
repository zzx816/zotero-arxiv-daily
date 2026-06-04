from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Pt
from loguru import logger

from .arxiv_email_parser import EmailPaper
from .paper_analyzer import PaperAnalysis


def write_report(
    papers: list[EmailPaper],
    analyses: list[PaperAnalysis],
    output_dir: str | Path,
    report_date: date | None = None,
    diagnostics: list[str] | None = None,
) -> Path:
    report_date = report_date or date.today()
    output_path = Path(output_dir) / f"{report_date:%Y-%m-%d}_paper_triage_report.docx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    _set_default_font(document)
    document.add_heading("每日 arXiv 论文迁移可行性分析报告", level=0)

    _add_overview(document, papers, analyses, report_date, diagnostics or [])
    _add_quick_table(document, papers, analyses)
    _add_details(document, papers, analyses)
    _add_reading_list(document, "今日精读清单", papers, analyses, "精读")
    _add_reading_list(document, "今日略读清单", papers, analyses, "略读")
    _add_reading_list(document, "今日跳过清单", papers, analyses, "跳过")
    _add_tags(document, analyses)
    _add_experiment_ideas(document, papers, analyses)

    document.save(output_path)
    logger.info("Wrote paper triage report to {}", output_path)
    return output_path


def _set_default_font(document: Document) -> None:
    style = document.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(10.5)


def _add_overview(
    document: Document,
    papers: list[EmailPaper],
    analyses: list[PaperAnalysis],
    report_date: date,
    diagnostics: list[str],
) -> None:
    document.add_heading("今日总览", level=1)
    document.add_paragraph(f"报告日期：{report_date:%Y-%m-%d}")
    document.add_paragraph(f"解析论文数量：{len(papers)}")
    counts = Counter(a.reading_recommendation for a in analyses)
    document.add_paragraph(
        f"阅读建议统计：精读 {counts.get('精读', 0)} 篇，"
        f"略读 {counts.get('略读', 0)} 篇，跳过 {counts.get('跳过', 0)} 篇。"
    )
    if diagnostics:
        document.add_paragraph("诊断信息：")
        for item in diagnostics:
            document.add_paragraph(item, style="List Bullet")


def _add_quick_table(document: Document, papers: list[EmailPaper], analyses: list[PaperAnalysis]) -> None:
    document.add_heading("快速筛选表", level=1)
    table = document.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    headers = ["标题", "评分", "方向", "迁移可行性", "阅读建议", "标签"]
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header

    if not papers:
        row = table.add_row().cells
        row[0].text = "未解析到论文"
        return

    for paper, analysis in zip(papers, analyses):
        row = table.add_row().cells
        row[0].text = paper.title
        row[1].text = f"{analysis.relevance_score:g}"
        row[2].text = analysis.matched_research_direction
        row[3].text = analysis.transfer_feasibility
        row[4].text = analysis.reading_recommendation
        row[5].text = ", ".join(analysis.keyword_tags)


def _add_details(document: Document, papers: list[EmailPaper], analyses: list[PaperAnalysis]) -> None:
    document.add_heading("逐篇论文详细分析", level=1)
    if not papers:
        document.add_paragraph("未解析到可分析论文。")
        return

    for index, (paper, analysis) in enumerate(zip(papers, analyses), start=1):
        document.add_heading(f"{index}. {paper.title}", level=2)
        document.add_paragraph(f"arXiv 链接：{paper.arxiv_url}")
        document.add_paragraph(f"相关性评分：{analysis.relevance_score:g}/10")
        document.add_paragraph(f"最匹配研究方向：{analysis.matched_research_direction}")
        document.add_paragraph(f"核心贡献：{analysis.core_contribution}")
        document.add_paragraph(f"方法类型：{analysis.method_type}")
        document.add_paragraph(f"技术要素：{_format_flags(analysis.technique_flags)}")
        document.add_paragraph(
            f"是否可迁移到水下目标识别：{'是' if analysis.transferable_to_underwater else '否'}"
        )
        document.add_paragraph(f"迁移可行性：{analysis.transfer_feasibility}")
        document.add_paragraph(f"可以迁移的部分：{analysis.transferable_parts}")
        document.add_paragraph(f"不容易迁移的部分：{analysis.hard_to_transfer_parts}")
        document.add_paragraph(f"DeepShip / ShipsEar 最小实验方案：{analysis.minimal_experiment}")
        document.add_paragraph(f"推荐阅读建议：{analysis.reading_recommendation}")
        document.add_paragraph(f"关键词标签：{', '.join(analysis.keyword_tags)}")
        if analysis.error:
            document.add_paragraph(f"自动分析错误：{analysis.error}")


def _add_reading_list(
    document: Document,
    heading: str,
    papers: list[EmailPaper],
    analyses: list[PaperAnalysis],
    recommendation: str,
) -> None:
    document.add_heading(heading, level=1)
    selected = [
        (paper, analysis)
        for paper, analysis in zip(papers, analyses)
        if analysis.reading_recommendation == recommendation
    ]
    if not selected:
        document.add_paragraph("无")
        return
    for paper, analysis in selected:
        document.add_paragraph(
            f"{paper.title}（{analysis.relevance_score:g}/10，{analysis.transfer_feasibility}）",
            style="List Bullet",
        )


def _add_tags(document: Document, analyses: list[PaperAnalysis]) -> None:
    document.add_heading("可加入 Zotero 的标签建议", level=1)
    tags = []
    for analysis in analyses:
        tags.extend(analysis.keyword_tags)
    unique_tags = sorted(set(tag for tag in tags if tag))
    document.add_paragraph(", ".join(unique_tags) if unique_tags else "无")


def _add_experiment_ideas(document: Document, papers: list[EmailPaper], analyses: list[PaperAnalysis]) -> None:
    document.add_heading("后续实验启发", level=1)
    selected = [
        (paper, analysis)
        for paper, analysis in zip(papers, analyses)
        if analysis.reading_recommendation in {"精读", "略读"}
    ]
    if not selected:
        document.add_paragraph("暂无明确实验启发。")
        return
    for paper, analysis in selected:
        document.add_paragraph(f"{paper.title}：{analysis.minimal_experiment}", style="List Bullet")


def _format_flags(flags: dict[str, bool]) -> str:
    positives = [name for name, enabled in flags.items() if enabled]
    return ", ".join(positives) if positives else "未明显涉及"

