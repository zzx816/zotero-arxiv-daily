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
      "matched_research_direction": "广义零样本学习 GZSL",
      "core_contribution": "提出一种生成式 GZSL 方法。",
      "method_type": "generative GZSL",
      "technique_flags": {
        "zero-shot": false,
        "few-shot": true,
        "GZSL": true,
        "semantic attribute embedding": true,
        "prototype learning": true,
        "generative feature synthesis (GAN/VAE)": true,
        "contrastive learning": false,
        "self-supervised pretraining": false,
        "domain adaptation": false,
        "domain generalization": false,
        "class imbalance / long-tail": false,
        "metric learning": false,
        "audio-text alignment (CLAP-like)": false,
        "LLM-generated semantic attributes": false,
        "incremental / continual learning": false,
        "open-set or OOD detection (次要,非主线)": false
      },
      "zsl_fsl_score": 9,
      "method_transfer_score": 8,
      "acoustic_modality_score": 6,
      "novelty_score": 8,
      "experiment_feasibility_score": 7,
      "risk_score": 2,
      "score_rationale": "GZSL 机制和生成式特征合成都有迁移价值。",
      "transferable_parts": "生成式特征合成和语义嵌入桥接。",
      "hard_to_transfer_parts": "视觉增强策略。",
      "minimal_experiment": "在 DeepShip 上构造 seen/unseen 划分并用 Log-Mel 特征验证。",
      "inspiration_note": "可以借鉴语义嵌入到特征生成的桥接方式。",
      "keyword_tags": ["GZSL", "prototype", "few-shot"]
    }
    """
    analyzer = PaperAnalyzer(
        api_key="fake",
        base_url="http://localhost",
        client=FakeClient(content),
    )

    analysis = analyzer.analyze_one(EmailPaper("A title", "An abstract", "https://arxiv.org/abs/1"))

    assert analysis.relevance_score == 8.0
    assert analysis.transfer_feasibility == "高"
    assert analysis.reading_recommendation == "精读"
    assert analysis.technique_flags["GZSL"] is True
    assert analysis.zsl_fsl_score == 9
    assert analysis.method_transfer_score == 8
    assert analysis.inspiration_note == "可以借鉴语义嵌入到特征生成的桥接方式。"


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
