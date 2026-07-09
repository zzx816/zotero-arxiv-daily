from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import OpenAI

from .arxiv_email_parser import EmailPaper


# 技术标签:覆盖你实际关心的方法族。open-set/OOD 保留但放在最后并标注"次要",
# 只用来标记论文是否涉及该技术,不代表会被优先检索或优先打分——
# GZSL 是当前主线,这一点由 paper_triage_config.yaml 的 research_directions 统一定义。
TECHNIQUE_FLAGS = [
    "zero-shot",
    "few-shot",
    "GZSL",
    "semantic attribute embedding",
    "prototype learning",
    "generative feature synthesis (GAN/VAE)",
    "contrastive learning",
    "self-supervised pretraining",
    "domain adaptation",
    "domain generalization",
    "class imbalance / long-tail",
    "metric learning",
    "audio-text alignment (CLAP-like)",
    "LLM-generated semantic attributes",
    "incremental / continual learning",
    "open-set or OOD detection (次要,非主线)",
]

# config.scoring_weights 缺失时的默认权重,五项之和会被归一化为 1.0
DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "zsl_fsl_score": 0.35,
    "method_transfer_score": 0.25,
    "acoustic_modality_score": 0.15,
    "novelty_score": 0.15,
    "experiment_feasibility_score": 0.10,
}

DEFAULT_SCORING_ANCHORS: dict[str, str] = {
    "8-10": "可直接用于设计 DeepShip / ShipsEar 的 GZSL 实验，或提供了尚未覆盖的语义构建/特征生成新思路",
    "6-7.9": "方法可迁移，但需要较大改造",
    "3-5.9": "概念相关，但没有 unseen 类识别或语义桥接的实验设计，参考价值弱",
    "0-2.9": "与水声分类、GZSL/FSL、类别语义构建几乎无关，或是纯 OSR/OOD 拒识且不含 unseen 类分类环节",
}

_SUB_SCORE_LABELS: dict[str, str] = {
    "zsl_fsl_score": "ZSL/FSL相关度",
    "method_transfer_score": "方法可迁移性",
    "acoustic_modality_score": "声学模态匹配度",
    "novelty_score": "新颖度/灵感值",
    "experiment_feasibility_score": "实验可行性",
}


@dataclass
class PaperAnalysis:
    relevance_score: float
    matched_research_direction: str
    core_contribution: str
    method_type: str
    score_rationale: str = ""
    zsl_fsl_score: float = 0.0
    method_transfer_score: float = 0.0
    acoustic_modality_score: float = 0.0
    novelty_score: float = 0.0
    experiment_feasibility_score: float = 0.0
    risk_score: float = 0.0
    decision_reason: str = ""
    technique_flags: dict[str, bool] = field(default_factory=dict)
    transferable_to_underwater: bool = False
    transfer_feasibility: str = "低"
    transferable_parts: str = ""
    hard_to_transfer_parts: str = ""
    minimal_experiment: str = ""
    inspiration_note: str = ""
    reading_recommendation: str = "跳过"
    keyword_tags: list[str] = field(default_factory=list)
    error_category: str | None = None
    error: str | None = None


class PaperAnalyzer:
    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        generation_kwargs: dict[str, Any] | None = None,
        research_directions: list[str] | None = None,
        judgment_criteria: list[str] | None = None,
        scoring_weights: dict[str, float] | None = None,
        scoring_anchors: dict[str, str] | None = None,
        provider: str = "OpenAI-compatible",
        key_source: str = "unknown",
        warnings: list[str] | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Set OPENAI_API_KEY or SILICONFLOW_API_KEY for paper triage analysis.")
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.generation_kwargs = {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 2500}
        self.generation_kwargs.update(
            {key: value for key, value in (generation_kwargs or {}).items() if value not in (None, "")}
        )
        self.research_directions = research_directions or []
        self.judgment_criteria = judgment_criteria or []
        self.scoring_weights = _normalize_weights(scoring_weights or DEFAULT_SCORING_WEIGHTS)
        self.scoring_anchors = scoring_anchors or DEFAULT_SCORING_ANCHORS
        self.config_summary = {
            "provider": provider,
            "base_url": base_url or "https://api.openai.com/v1",
            "model": self.generation_kwargs.get("model", "gpt-4o-mini"),
            "key_source": key_source,
            "warnings": warnings or [],
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PaperAnalyzer":
        llm_config = config.get("llm", {})
        siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if siliconflow_key:
            api_key = siliconflow_key
            key_source = "SILICONFLOW_API_KEY"
            provider = "SiliconFlow"
        elif openai_key:
            api_key = openai_key
            key_source = "OPENAI_API_KEY"
            provider = "OpenAI-compatible"
        else:
            api_key = llm_config.get("api_key", "")
            key_source = "config.llm.api_key"
            provider = "OpenAI-compatible"

        base_url = os.getenv("OPENAI_API_BASE") or llm_config.get("api_base") or None
        model = os.getenv("PAPER_TRIAGE_MODEL")
        generation_kwargs = dict(llm_config.get("generation_kwargs", {}))
        if model:
            generation_kwargs["model"] = model
        selected_model = generation_kwargs.get("model")
        warnings = []
        if base_url and "siliconflow" in base_url.lower():
            provider = "SiliconFlow"
            if not selected_model or selected_model == "gpt-4o-mini":
                warnings.append(
                    "SiliconFlow base URL is configured, but PAPER_TRIAGE_MODEL is empty or still gpt-4o-mini. "
                    "Set PAPER_TRIAGE_MODEL to a SiliconFlow chat model."
                )
        return cls(
            api_key=api_key,
            base_url=base_url,
            generation_kwargs=generation_kwargs,
            research_directions=list(config.get("research_directions", [])),
            judgment_criteria=list(config.get("judgment_criteria", [])),
            scoring_weights=dict(config.get("scoring_weights", {})) or None,
            scoring_anchors=dict(config.get("scoring_anchors", {})) or None,
            provider=provider,
            key_source=key_source,
            warnings=warnings,
        )

    def analyze(self, papers: list[EmailPaper]) -> list[PaperAnalysis]:
        analyses = []
        for paper in papers:
            analyses.append(self.analyze_one(paper))
        return analyses

    def analyze_one(self, paper: EmailPaper) -> PaperAnalysis:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是严谨的科研论文迁移可行性分析助手。只基于用户提供的标题和摘要判断，"
                            "不要假设已阅读 PDF 或全文。只输出分项评分和分析文字，不要自己计算综合分，"
                            "也不要给出精读/略读/跳过的最终结论——这两项由调用方根据分项分数统一计算，"
                            "以保证每天的评分口径一致。必须输出一个 JSON 对象，不要输出其它文字。"
                        ),
                    },
                    {"role": "user", "content": self._build_prompt(paper)},
                ],
                **self.generation_kwargs,
            )
        except Exception as exc:
            category = _classify_api_exception(exc)
            logger.warning("Failed to call LLM for '{}': {}: {}", paper.title, category, exc)
            return _fallback_analysis(paper, category, str(exc))

        content = response.choices[0].message.content or ""
        try:
            data = json.loads(_extract_json(content))
            return _analysis_from_dict(data, self.scoring_weights)
        except Exception as exc:
            category = "JSON 解析失败"
            logger.warning("Failed to parse LLM JSON for '{}': {}", paper.title, exc)
            return _fallback_analysis(paper, category, f"{exc}; response preview: {content[:300]}")

    def _build_prompt(self, paper: EmailPaper) -> str:
        directions = "\n".join(f"- {item}" for item in self.research_directions)
        criteria = "\n".join(f"- {item}" for item in self.judgment_criteria)
        anchors = "\n".join(f"- {k} 分：{v}" for k, v in self.scoring_anchors.items())
        flags = ", ".join(TECHNIQUE_FLAGS)
        weight_hint = "、".join(f"{_SUB_SCORE_LABELS.get(k, k)} {v:.0%}" for k, v in self.scoring_weights.items())
        return f"""
请分析下面论文是否值得用于"水下目标识别 / 广义零样本 GZSL / 少样本 FSL"方向的迁移研究。
核心问题只有一个：这篇论文的方法，能不能让模型正确分类"训练时没见过的类别"？

研究方向（按重要性排列，标【暂缓，非主线】的不要给高分）：
{directions}

判断标准：
{criteria}

分项评分锚点（每一项分项分都参考这个尺度，0-10 之间可以有小数）：
{anchors}

论文：
标题：{paper.title}
摘要或邮件摘要：{paper.abstract or "未提供"}
链接：{paper.arxiv_url}

请只返回 JSON，不要 Markdown 代码块，不要输出 relevance_score 或 reading_recommendation
（这两项由调用方按权重 {weight_hint} 从下面几项分项分统一计算，避免每次口径不一致）。字段必须为：
{{
  "matched_research_direction": "最匹配的研究方向，如果是【暂缓，非主线】方向请明确写出",
  "core_contribution": "核心贡献，1-2句话",
  "method_type": "方法类型，例如 embedding-based GZSL / generative GZSL / OSR / 通用自监督表征 等",
  "technique_flags": {{"zero-shot": true/false, ...}},
  "zsl_fsl_score": 0-10 的数字,
  "method_transfer_score": 0-10 的数字,
  "acoustic_modality_score": 0-10 的数字,
  "novelty_score": 0-10 的数字,
  "experiment_feasibility_score": 0-10 的数字,
  "risk_score": 0-10 的数字（风险，越高越差，例如严重依赖大规模标注、复现成本高、依赖过时环境）,
  "score_rationale": "1-3句话说明上面几个分项分数为什么这样打",
  "transferable_parts": "可以迁移的部分",
  "hard_to_transfer_parts": "不容易迁移的部分",
  "minimal_experiment": "如果迁移到 DeepShip / ShipsEar，最小实验方案",
  "inspiration_note": "这篇论文有没有值得借鉴到你自己 GZSL 课题里的新颖设计（新的语义构建方式/损失函数/评估协议/数据增强等），具体写1-2句，没有就写「无明显新颖点」",
  "keyword_tags": ["3-5个关键词标签"]
}}

technique_flags 必须覆盖这些键：{flags}。
""".strip()


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """只保留已知的五个分项键，缺失的补 0，然后归一化到总和为 1.0。"""
    clean = {key: max(0.0, float(weights.get(key, 0.0))) for key in DEFAULT_SCORING_WEIGHTS}
    total = sum(clean.values())
    if total <= 0:
        return dict(DEFAULT_SCORING_WEIGHTS)
    return {key: value / total for key, value in clean.items()}


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return text[start : end + 1]


def _analysis_from_dict(data: dict[str, Any], scoring_weights: dict[str, float]) -> PaperAnalysis:
    flags = data.get("technique_flags") or {}
    normalized_flags = {name: bool(flags.get(name, False)) for name in TECHNIQUE_FLAGS}
    tags = data.get("keyword_tags") or []
    tags = [str(tag).strip() for tag in tags if str(tag).strip()][:5]

    sub_scores = {key: _clamp_score(data.get(key, 0)) for key in scoring_weights}
    relevance_score = round(sum(sub_scores[key] * weight for key, weight in scoring_weights.items()), 1)
    risk_score = _clamp_score(data.get("risk_score", 0))

    method_transfer_score = sub_scores.get("method_transfer_score", 0.0)
    transfer_feasibility = _feasibility_from_score(method_transfer_score)
    reading_recommendation = _recommendation_from_score(relevance_score)
    decision_reason = _compose_decision_reason(sub_scores, scoring_weights, relevance_score, reading_recommendation, risk_score)

    return PaperAnalysis(
        relevance_score=relevance_score,
        matched_research_direction=str(data.get("matched_research_direction") or "未明确"),
        core_contribution=str(data.get("core_contribution") or "LLM 未给出核心贡献。"),
        method_type=str(data.get("method_type") or "未明确"),
        score_rationale=str(data.get("score_rationale") or "未明确"),
        zsl_fsl_score=sub_scores.get("zsl_fsl_score", 0.0),
        method_transfer_score=method_transfer_score,
        acoustic_modality_score=sub_scores.get("acoustic_modality_score", 0.0),
        novelty_score=sub_scores.get("novelty_score", 0.0),
        experiment_feasibility_score=sub_scores.get("experiment_feasibility_score", 0.0),
        risk_score=risk_score,
        decision_reason=decision_reason,
        technique_flags=normalized_flags,
        transferable_to_underwater=method_transfer_score >= 4.0,
        transfer_feasibility=transfer_feasibility,
        transferable_parts=str(data.get("transferable_parts") or "未明确"),
        hard_to_transfer_parts=str(data.get("hard_to_transfer_parts") or "未明确"),
        minimal_experiment=str(data.get("minimal_experiment") or "未明确"),
        inspiration_note=str(data.get("inspiration_note") or "无明显新颖点"),
        reading_recommendation=reading_recommendation,
        keyword_tags=tags or ["待复核"],
    )


def _feasibility_from_score(score: float) -> str:
    if score >= 7:
        return "高"
    if score >= 4:
        return "中"
    return "低"


def _recommendation_from_score(score: float) -> str:
    if score >= 8:
        return "精读"
    if score >= 5:
        return "略读"
    return "跳过"


def _compose_decision_reason(
    sub_scores: dict[str, float],
    scoring_weights: dict[str, float],
    relevance_score: float,
    reading_recommendation: str,
    risk_score: float,
) -> str:
    parts = [
        f"{_SUB_SCORE_LABELS.get(key, key)} {value:g}(权重{scoring_weights[key]:.0%})"
        for key, value in sub_scores.items()
    ]
    reason = f"综合分 {relevance_score:g}/10 = " + " + ".join(parts) + f" → {reading_recommendation}"
    if risk_score >= 7:
        reason += f"；风险提示：risk_score {risk_score:g}/10 偏高，注意复现/迁移成本"
    return reason


def _fallback_analysis(paper: EmailPaper, error_category: str, error: str) -> PaperAnalysis:
    abstract_preview = paper.abstract[:180] if paper.abstract else "邮件中未提供摘要或 TLDR。"
    safe_error = _sanitize_error(error)
    return PaperAnalysis(
        relevance_score=0,
        matched_research_direction="待人工复核",
        core_contribution=f"自动分析失败。摘要片段：{abstract_preview}",
        method_type="待人工复核",
        score_rationale="LLM 调用或解析失败，无分项评分。",
        technique_flags={name: False for name in TECHNIQUE_FLAGS},
        transferable_to_underwater=False,
        transfer_feasibility="低",
        transferable_parts="待人工复核",
        hard_to_transfer_parts="LLM 分析失败，无法可靠判断。",
        minimal_experiment="待人工复核后再设计实验。",
        inspiration_note="待人工复核",
        decision_reason="LLM 分析失败，归为「跳过」以避免误导，请人工复核原文。",
        reading_recommendation="跳过",
        keyword_tags=["分析失败", "待复核"],
        error_category=error_category,
        error=f"{error_category}: {safe_error}",
    )


def _classify_api_exception(exc: Exception) -> str:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(marker in text for marker in ["401", "unauthorized", "authentication", "invalid api key", "api key"]):
        return "认证失败"
    if any(marker in text for marker in ["model_not_found", "model not found", "does not exist", "invalid model"]):
        return "模型不存在"
    if any(marker in text for marker in ["429", "rate limit", "too many requests"]):
        return "请求限流"
    if any(marker in text for marker in ["quota", "insufficient", "balance", "billing", "payment"]):
        return "额度或余额不足"
    if any(marker in text for marker in ["timeout", "connection", "network", "connecterror", "readtimeout"]):
        return "网络错误"
    return "LLM API 调用失败"


def _sanitize_error(error: str) -> str:
    text = str(error)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", text)
    text = re.sub(r"GOCSPX-[A-Za-z0-9_\-]+", "GOCSPX-***", text)
    return text[:1000]


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(10.0, max(0.0, score))


def _normalize_choice(value: Any, choices: list[str], default: str) -> str:
    text = str(value or "")
    for choice in choices:
        if choice in text:
            return choice
    return default
