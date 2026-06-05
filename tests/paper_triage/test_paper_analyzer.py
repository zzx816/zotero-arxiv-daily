from types import SimpleNamespace

from paper_triage.arxiv_email_parser import EmailPaper
from paper_triage.paper_analyzer import PaperAnalyzer


class FakeCompletions:
    def __init__(self, content: str = "", exc: Exception | None = None):
        self.content = content
        self.exc = exc

    def create(self, **kwargs):
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content: str = "", exc: Exception | None = None):
        self.chat = SimpleNamespace(completions=FakeCompletions(content, exc))


def test_analyzer_parses_json_response():
    content = """
    {
      "relevance_score": 8.5,
      "matched_research_direction": "开放集识别 OSR",
      "core_contribution": "提出一种开放集原型方法。",
      "method_type": "prototype learning",
      "technique_flags": {
        "zero-shot": false,
        "few-shot": true,
        "GZSL": false,
        "open-set": true,
        "OOD": true,
        "domain adaptation": false,
        "semantic embedding": false,
        "prototype learning": true,
        "contrastive learning": false
      },
      "transferable_to_underwater": true,
      "transfer_feasibility": "高",
      "transferable_parts": "原型分类头和未知类阈值。",
      "hard_to_transfer_parts": "视觉增强策略。",
      "minimal_experiment": "在 DeepShip 上用 Log-Mel 特征训练原型网络。",
      "reading_recommendation": "精读",
      "keyword_tags": ["OSR", "prototype", "few-shot"]
    }
    """
    analyzer = PaperAnalyzer(
        api_key="fake",
        base_url="http://localhost",
        client=FakeClient(content),
    )

    analysis = analyzer.analyze_one(EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1"))

    assert analysis.relevance_score == 8.5
    assert analysis.transfer_feasibility == "高"
    assert analysis.reading_recommendation == "精读"
    assert analysis.technique_flags["open-set"] is True


def test_analyzer_falls_back_on_invalid_json():
    analyzer = PaperAnalyzer(
        api_key="fake",
        base_url="http://localhost",
        client=FakeClient("not json"),
    )

    analysis = analyzer.analyze_one(EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1"))

    assert analysis.error is not None
    assert analysis.error_category == "JSON 解析失败"
    assert analysis.reading_recommendation == "跳过"
    assert analysis.keyword_tags == ["分析失败", "待复核"]


def test_analyzer_ignores_empty_generation_kwargs():
    analyzer = PaperAnalyzer(
        api_key="fake",
        base_url="http://localhost",
        generation_kwargs={"model": "", "temperature": None},
        client=FakeClient("{}"),
    )

    assert analyzer.generation_kwargs["model"] == "gpt-4o-mini"
    assert analyzer.generation_kwargs["temperature"] == 0.2


def test_from_config_prefers_siliconflow_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "siliconflow-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("PAPER_TRIAGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    analyzer = PaperAnalyzer.from_config({"llm": {"generation_kwargs": {"model": "gpt-4o-mini"}}})

    assert analyzer.config_summary["provider"] == "SiliconFlow"
    assert analyzer.config_summary["key_source"] == "SILICONFLOW_API_KEY"
    assert analyzer.config_summary["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert analyzer.config_summary["warnings"] == []


def test_from_config_warns_about_default_model_with_siliconflow(monkeypatch):
    monkeypatch.delenv("PAPER_TRIAGE_MODEL", raising=False)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "siliconflow-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")

    analyzer = PaperAnalyzer.from_config({"llm": {"generation_kwargs": {"model": "gpt-4o-mini"}}})

    assert analyzer.config_summary["warnings"]


def test_analyzer_classifies_api_errors_and_masks_secrets():
    analyzer = PaperAnalyzer(
        api_key="fake",
        base_url="http://localhost",
        client=FakeClient(exc=RuntimeError("401 Unauthorized invalid API key sk-1234567890abcdef")),
    )

    analysis = analyzer.analyze_one(EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1"))

    assert analysis.error_category == "认证失败"
    assert "sk-1234567890abcdef" not in analysis.error
    assert "sk-***" in analysis.error
