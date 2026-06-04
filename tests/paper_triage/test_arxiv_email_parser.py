from paper_triage.arxiv_email_parser import normalize_arxiv_url, parse_arxiv_email


def _sample_html(count: int = 6) -> str:
    blocks = []
    for idx in range(count):
        blocks.append(
            f"""
            <table>
              <tr><td style="font-size:20px;font-weight:bold;">Paper Title {idx} for Few Shot Open Set</td></tr>
              <tr><td>Author A, Author B</td></tr>
              <tr><td><strong>Relevance:</strong> {7 + idx / 10}</td></tr>
              <tr><td><strong>TLDR:</strong> This paper studies prototype learning for open-set recognition.</td></tr>
              <tr><td><a href="https://arxiv.org/pdf/2401.0000{idx}.pdf">PDF</a></td></tr>
            </table>
            """
        )
    return "<html><body>" + "\n".join(blocks) + "</body></html>"


def test_parse_zotero_arxiv_daily_html_limits_to_five():
    papers = parse_arxiv_email(_sample_html(), max_papers=5)

    assert len(papers) == 5
    assert papers[0].title == "Paper Title 0 for Few Shot Open Set"
    assert papers[0].abstract == "This paper studies prototype learning for open-set recognition."
    assert papers[0].arxiv_url == "https://arxiv.org/abs/2401.00000"


def test_normalize_arxiv_pdf_url_to_abs_url():
    assert normalize_arxiv_url("https://arxiv.org/pdf/2310.12345v2.pdf") == "https://arxiv.org/abs/2310.12345v2"
    assert normalize_arxiv_url("https://arxiv.org/abs/2310.12345") == "https://arxiv.org/abs/2310.12345"

