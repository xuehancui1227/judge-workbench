import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from judgekit.models import Concept, Plan, Rule
from judgekit.plans import orphan_rules, revise_plan, update_concepts, validate_mapping


def base_plan():
    return Plan(evaluation_target="回答质量", original_rules="原始评分规则",
                concepts=[Concept(key="answer", description="待评回答", source="column", column_ids=["c1"])],
                rules=[Rule(name="not_empty", concept="answer", operator="nonempty")])


def test_delete_linked_field_requires_explicit_consent():
    plan = base_plan()
    with pytest.raises(ValueError, match="删除的字段仍被规则引用"):
        update_concepts(plan, [])
    assert len(plan.concepts) == len(plan.rules) == 1
    updated = update_concepts(plan, [], remove_linked_rules=True)
    assert updated.rules == updated.concepts == []


def test_fields_add_describe_and_duplicate_validation():
    plan = base_plan()
    changed = plan.concepts[0].model_copy(update={"description": "最终答复"})
    updated = update_concepts(plan, [changed, Concept(key="vlm", description="视觉证据")])
    assert updated.concepts[0].description == "最终答复"
    assert len(updated.concepts) == 2 and len(plan.concepts) == 1
    with pytest.raises(ValueError):
        update_concepts(plan, [changed, changed])


def test_orphan_rule_is_identified_not_silently_removed():
    plan = base_plan()
    plan.rules.append(Rule(name="vlm_use_enum", concept="unknown", operator="one_of", expected='["是", "否"]'))
    assert [(i, r.name) for i, r in orphan_rules(plan)] == [(1, "vlm_use_enum")]
    with pytest.raises(ValueError, match="引用了不存在的概念"):
        validate_mapping(plan, {"c1": "answer"})


def test_natural_language_revision_preserves_original_and_uses_current_schema():
    plan = base_plan()
    seen = []

    class Client:
        config = SimpleNamespace(mock=False, api_key="private-test-key")

        def complete(self, system, payload, schema):
            seen.append(payload)
            result = plan.model_dump()
            result["name"] = "新方案"
            result["original_rules"] = "模型不应覆盖溯源规则"
            result["judge_prompt"] = "最新修改：不评价忠实性"
            return result, {}

    updated = revise_plan(plan, "不评价忠实性", [{"id": "c1", "name": "回答"}], Client())
    assert updated.name == "新方案" and plan.name != updated.name
    assert updated.original_rules == plan.original_rules
    assert seen[0]["change_request"] == "不评价忠实性"
    assert seen[0]["existing_plan"] == plan.model_dump()
    assert "schema" in seen[0] and "api_key" not in seen[0]


def demo_app(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"), default_timeout=30).run()
    next(c for c in app.checkbox if c.label == "加载内置演示数据").check().run()
    next(b for b in app.button if b.label == "理解规则并推荐字段").click().run()
    assert not app.exception
    return app


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_ui_add_describe_delete_fields(tmp_path, monkeypatch):
    app = demo_app(tmp_path, monkeypatch)
    next(x for x in app.text_input if x.label == "新字段标识（英文、数字、下划线）").set_value("new_evidence")
    next(x for x in app.text_input if x.label == "新字段描述").set_value("新增参考证据")
    button(app, "添加字段").click().run()
    assert not app.exception
    plan = Plan.model_validate_json(app.session_state["plan_draft"])
    assert plan.concepts[-1].key == "new_evidence"
    descriptions = [x for x in app.text_input if x.label == "字段描述"]
    descriptions[-1].set_value("更新后的证据描述")
    button(app, "应用字段修改").click().run()
    assert Plan.model_validate_json(app.session_state["plan_draft"]).concepts[-1].description == "更新后的证据描述"
    deletions = [x for x in app.checkbox if x.label == "删除此字段"]
    deletions[-1].check()
    button(app, "应用字段修改").click().run()
    assert not app.exception
    assert len(Plan.model_validate_json(app.session_state["plan_draft"]).concepts) == 2


def test_ui_orphan_can_be_rebound_then_confirmed(tmp_path, monkeypatch):
    app = demo_app(tmp_path, monkeypatch)
    plan = Plan.model_validate_json(app.session_state["plan_draft"])
    plan.rules = [Rule(name="vlm_use_enum", concept="missing_vlm", operator="nonempty")]
    app.session_state["plan_draft"] = plan.model_dump_json()
    app.run()
    assert not app.exception
    assert button(app, "确认此方案").disabled
    control = next(x for x in app.selectbox if x.label.startswith("规则 vlm_use_enum"))
    control.select("answer").run()
    button(app, "修复并应用方案").click().run()
    assert not app.exception
    fixed = Plan.model_validate_json(app.session_state["plan_draft"])
    assert fixed.rules[0].concept == "answer"
    assert not button(app, "确认此方案").disabled


def test_ui_natural_revision_preview_apply_and_reconfirm(tmp_path, monkeypatch):
    import judgekit.plan_editor
    seen = []

    def fake_revision(plan, instruction, profile, client):
        seen.append(instruction)
        updated = plan.model_copy(deep=True)
        updated.name = "修改后的方案"
        return updated

    monkeypatch.setattr(judgekit.plan_editor, "revise_plan", fake_revision)
    app = demo_app(tmp_path, monkeypatch)
    button(app, "确认此方案").click().run()
    assert "confirmed" in app.session_state
    next(x for x in app.text_area if x.label == "告诉模型如何修改").set_value("修改方案名称")
    next(x for x in app.checkbox if x.label == "允许将当前方案、字段样例和修改要求发送到配置的API").check()
    button(app, "生成修改建议").click().run()
    assert not app.exception
    assert Plan.model_validate_json(app.session_state["plan_draft"]).name != "修改后的方案"
    assert seen == ["修改方案名称"]
    button(app, "应用修改后的方案").click().run()
    assert not app.exception
    assert Plan.model_validate_json(app.session_state["plan_draft"]).name == "修改后的方案"
    assert "confirmed" not in app.session_state
