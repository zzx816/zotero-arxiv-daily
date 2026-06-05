from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import OpenAI

from .arxiv_email_parser import EmailPaper


TECHNIQUE_FLAGS = [
    "zero-shot",
    "few-shot",
    "GZSL",
    "open-set",
    "OOD",
    "domain adaptation",
    "semantic embedding",
    "prototype learning",
    "contrastive learning",
]


@dataclass
class PaperAnalysis:
    relevance_score: float
    matched_research_direction: str
    core_contribution: str
    method_type: str
    technique_flags: dict[str, bool] = field(default_factory=dict)
    transferable_to_underwater: bool = False
    transfer_feasibility: str = "低"
    transferable_parts: str = ""
    hard_to_transfer_parts: str = ""
    minimal_experiment: str = ""
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
        provider: str = "OpenAI-compatible",
        key_source: str = "unknown",
        warnings: list[str] | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Set OPENAI_API_KEY or SILICONFLOW_API_KEY for paper triage analysis.")
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.generation_kwargs = {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 2000}
        self.generation_kwargs.update(
            {key: value for key, value in (generation_kwargs or {}).items() if value not in (None, "")}
        )
        self.research_directions = research_directions or []
        self.judgment_criteria = judgment_criteria or []
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
                            "不要假设已阅读 PDF 或全文。必须输出一个 JSON 对象。"
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
            return _analysis_from_dict(json.loads(_extract_json(content)))
        except Exception as exc:
            category = "JSON 解析失败"
            logger.warning("Failed to parse LLM JSON for '{}': {}", paper.title, exc)
            return _fallback_analysis(paper, category, f"{exc}; response preview: {content[:300]}")

    def _build_prompt(self, paper: EmailPaper) -> str:
        directions = "\n".join(f"- {item}" for item in self.research_directions)
        criteria = "\n".join(f"- {item}" for item in self.judgment_criteria)
        flags = ", ".join(TECHNIQUE_FLAGS)
        return f"""
请分析下面论文是否值得用于“水下目标识别 / 零样本 / 少样本 / 开放集识别”方向的迁移研究。

研究方向：
{directions}

判断标准：
{criteria}

论文：
标题：{paper.title}
摘要或邮件摘要：{paper.abstract or "未提供"}
链接：{paper.arxiv_url}

请只返回 JSON，不要 Markdown。字段必须为：
{{
  "relevance_score": 0-10 的数字,
  "matched_research_direction": "最匹配的研究方向",
  "core_contribution": "核心贡献，1-2句话",
  "method_type": "方法类型",
  "technique_flags": {{"zero-shot": true/false, ...}},
  "transferable_to_underwater": true/false,
  "transfer_feasibility": "高/中/低",
  "transferable_parts": "可以迁移的部分",
  "hard_to_transfer_parts": "不容易迁移的部分",
  "minimal_experiment": "如果迁移到 DeepShip / ShipsEar，最小实验方案",
  "reading_recommendation": "精读/略读/跳过",
  "keyword_tags": ["3-5个关键词标签"]
}}

technique_flags 必须覆盖这些键：{flags}。
""".strip()


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return text[start : end + 1]


def _analysis_from_dict(data: dict[str, Any]) -> PaperAnalysis:
    flags = data.get("technique_flags") or {}
    normalized_flags = {name: bool(flags.get(name, False)) for name in TECHNIQUE_FLAGS}
    tags = data.get("keyword_tags") or []
    tags = [str(tag).strip() for tag in tags if str(tag).strip()][:5]

    return PaperAnalysis(
        relevance_score=_clamp_score(data.get("relevance_score", 0)),
        matched_research_direction=str(data.get("matched_research_direction") or "未明确"),
        core_contribution=str(data.get("core_contribution") or "LLM 未给出核心贡献。"),
        method_type=str(data.get("method_type") or "未明确"),
        technique_flags=normalized_flags,
        transferable_to_underwater=bool(data.get("transferable_to_underwater", False)),
        transfer_feasibility=_normalize_choice(data.get("transfer_feasibility"), ["高", "中", "低"], "低"),
        transferable_parts=str(data.get("transferable_parts") or "未明确"),
        hard_to_transfer_parts=str(data.get("hard_to_transfer_parts") or "未明确"),
        minimal_experiment=str(data.get("minimal_experiment") or "未明确"),
        reading_recommendation=_normalize_choice(data.get("reading_recommendation"), ["精读", "略读", "跳过"], "跳过"),
        keyword_tags=tags or ["待复核"],
    )


def _fallback_analysis(paper: EmailPaper, error_category: str, error: str) -> PaperAnalysis:
    abstract_preview = paper.abstract[:180] if paper.abstract else "邮件中未提供摘要或 TLDR。"
    safe_error = _sanitize_error(error)
    return PaperAnalysis(
        relevance_score=0,
        matched_research_direction="待人工复核",
        core_contribution=f"自动分析失败。摘要片段：{abstract_preview}",
        method_type="待人工复核",
        technique_flags={name: False for name in TECHNIQUE_FLAGS},
        transferable_to_underwater=False,
        transfer_feasibility="低",
        transferable_parts="待人工复核",
        hard_to_transfer_parts="LLM 分析失败，无法可靠判断。",
        minimal_experiment="待人工复核后再设计实验。",
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
        return 0
    return min(10, max(0, score))


def _normalize_choice(value: Any, choices: list[str], default: str) -> str:
    text = str(value or "")
    for choice in choices:
        if choice in text:
            return choice
    return default
