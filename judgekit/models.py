from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Concept(StrictModel):
    key: str
    description: str
    required: bool = True
    source: Literal["column", "constant", "not_applicable", "missing"] = "missing"
    column_ids: list[str] = Field(default_factory=list)
    constant: str = ""
    confidence: float = Field(default=0, ge=0, le=1)
    explanation: str = ""


class Dimension(StrictModel):
    key: str
    name: str
    description: str
    kind: Literal["number", "string", "boolean"] = "number"
    allowed_values: list[str | float | bool] = Field(default_factory=list)
    rubric: str = ""
    required: bool = True


class Rule(StrictModel):
    name: str
    concept: str
    operator: Literal["nonempty", "equals", "one_of", "contains", "json_valid"]
    expected: str = ""
    severity: Literal["review", "fail"] = "review"


class Plan(StrictModel):
    name: str = "未命名评测"
    version: str = "1.0"
    evaluation_target: str
    original_rules: str
    concepts: list[Concept]
    dimensions: list[Dimension] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    hard_fail_rules: list[str] = Field(default_factory=list)
    pass_condition: str = "仅按用户确认的规则判断；未定义通过标准时 overall_pass=null。"
    reference_required: bool = False
    judge_prompt: str = ""
    # A plan is data, never executable code. Uncompiled conditions remain LLM criteria.
    output_notes: str = ""

    @model_validator(mode="after")
    def unique_names(self):
        for items in (self.concepts, self.dimensions):
            keys = [x.key for x in items]
            if len(keys) != len(set(keys)):
                raise ValueError("概念和维度的 key 各自必须唯一")
        names = [x.name for x in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("评分维度名称必须唯一")
        reserved = {"status", "model", "rule_name", "rule_version", "time", "overall_pass", "reason",
                    "evidence", "error_type", "needs_review", "reference_sources", "raw_json",
                    "input_tokens", "output_tokens", "cost", "judgeable", "critical_error"}
        if any(d.name in reserved for d in self.dimensions):
            raise ValueError("维度名与公共结果列冲突，请改用业务维度名称")
        return self


class APIConfig(BaseModel):
    provider: str = "OpenAI兼容"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key: str = Field(default="", repr=False, exclude=True)
    mode: Literal["json_schema", "json_object", "plain"] = "json_object"
    temperature: float | None = 0
    timeout: float = Field(default=60, ge=1, le=600)
    retries: int = Field(default=2, ge=0, le=5)
    concurrency: int = Field(default=2, ge=1, le=16)
    rpm: int = Field(default=60, ge=1, le=6000)
    max_chars: int = Field(default=60000, ge=1000, le=500000)
    max_output_tokens: int = Field(default=4000, ge=256, le=32000)
    token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    input_price: float = Field(default=0, ge=0)
    output_price: float = Field(default=0, ge=0)
    mock: bool = False


def result_schema(plan: Plan) -> dict:
    dimensions = {}
    for d in plan.dimensions:
        schema = {"type": [d.kind, "null"]}
        if d.allowed_values:
            schema["enum"] = [*d.allowed_values, None]
        dimensions[d.key] = schema
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "judgeable": {"type": "boolean"},
            "overall_pass": {"type": ["boolean", "null"]},
            "reason": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "error_types": {"type": "array", "items": {"type": "string"}},
            "critical_error": {"type": "boolean"},
            "needs_review": {"type": "boolean"},
            "reference_ids": {"type": "array", "items": {"type": "string"}},
            "scores": {"type": "object", "properties": dimensions,
                       "required": list(dimensions), "additionalProperties": False},
        },
        "required": ["judgeable", "overall_pass", "reason", "evidence", "error_types",
                     "critical_error", "needs_review", "reference_ids", "scores"],
    }


def output_columns(plan: Plan) -> list[str]:
    common = ["judge_status", "judge_model", "judge_rule_name", "judge_rule_version", "judge_time",
              "judge_overall_pass", "judge_reason", "judge_evidence", "judge_error_type", "judge_needs_review",
              "judge_reference_sources", "judge_raw_json", "judge_input_tokens", "judge_output_tokens", "judge_cost",
              "judge_judgeable", "judge_critical_error"]
    return common+["judge_"+d.name for d in plan.dimensions]
