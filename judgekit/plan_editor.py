"""Session-local plan editing. No API calls until the user explicitly requests one."""
import json
import re

import streamlit as st

from .llm import CallFailure, Client
from .models import Concept, Plan
from .plans import orphan_rules, revise_plan, update_concepts, validate_mapping
from .storage import digest


def apply_draft(plan):
    st.session_state["plan_draft"] = plan.model_dump_json(indent=2)
    st.session_state.pop("confirmed", None)
    st.session_state.pop("revision_candidate", None)


def edit_fields(plan, ds):
    fingerprint = digest(plan.model_dump())[:12]
    with st.expander("调整字段对应（可增删字段、修改描述）", expanded=True):
        st.caption("这里的字段是评测方案所需的信息，不会新增、删除或修改上传文件的原始列。")
        edited, deleted = [], []
        for c in plan.concepts:
            key = f"field_{fingerprint}_{c.key}"
            a, b, d = st.columns([2, 2, 4])
            a.caption(f"标识：{c.key}")
            description = a.text_input("字段描述", c.description, key=key+"_description")
            required = a.checkbox("本次必需", value=c.required, key=key+"_required")
            remove = a.checkbox("删除此字段", key=key+"_delete")
            sources = ["column", "constant", "not_applicable", "missing"]
            src = b.selectbox("信息来源", sources, index=sources.index(c.source), key=key+"_source")
            cols, const = c.column_ids, c.constant
            if src == "column" and ds:
                cols = d.multiselect("对应列（可多选）", list(ds.columns),
                                     default=[k for k in cols if k in ds.columns],
                                     format_func=lambda k: f"{k} · {ds.columns[k]}", key=key+"_columns")
            elif src == "constant":
                const = d.text_input("固定内容", const, key=key+"_constant")
            if remove:
                deleted.append(c.key)
            else:
                edited.append(Concept(**{**c.model_dump(), "description": description,
                                         "required": required, "source": src,
                                         "column_ids": cols if src == "column" else [],
                                         "constant": const if src == "constant" else ""}))
        linked = [r.name for r in plan.rules if r.concept in deleted]
        remove_rules = False
        if linked:
            st.warning("这些字段被以下规则引用："+"、".join(linked))
            remove_rules = st.checkbox("确认同时删除上述关联规则（可能改变评测口径）", key="remove_rules_"+fingerprint)
        if st.button("应用字段修改"):
            try:
                if any(not c.description.strip() for c in edited):
                    raise ValueError("字段描述不能为空")
                updated = update_concepts(plan, edited, remove_linked_rules=remove_rules)
                apply_draft(updated)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        st.markdown("**添加评测字段**")
        a, b = st.columns(2)
        new_key = a.text_input("新字段标识（英文、数字、下划线）", key="new_key_"+fingerprint)
        new_description = b.text_input("新字段描述", key="new_description_"+fingerprint)
        st.caption("添加后默认标记为“信息尚缺”，再选择对应原始列或固定内容。")
        if st.button("添加字段"):
            try:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", new_key.strip()):
                    raise ValueError("字段标识须以英文字母或下划线开头，仅含英文、数字、下划线")
                if not new_description.strip():
                    raise ValueError("请填写新字段描述")
                updated = update_concepts(plan, plan.concepts+[
                    Concept(key=new_key.strip(), description=new_description.strip(), source="missing")])
                apply_draft(updated)
                st.rerun()
            except ValueError as error:
                st.error(str(error))


def repair_references(plan):
    broken = orphan_rules(plan)
    if not broken:
        return
    with st.expander("修复规则引用后应用方案", expanded=True):
        st.warning("规则引用了不存在的字段。请选择重新绑定字段，或明确删除该规则；不会自动忽略错误。")
        replacements = {}
        fingerprint = digest(plan.model_dump())[:12]
        options = ["保持待修复", "删除此规则"]+[c.key for c in plan.concepts]
        names = {c.key: f"{c.description}（{c.key}）" for c in plan.concepts}
        for i, rule in broken:
            replacements[i] = st.selectbox(
                f"规则 {rule.name}：原引用 {rule.concept}", options,
                format_func=lambda k: names.get(k, k), key=f"repair_{fingerprint}_{i}")
        deleting = any(v == "删除此规则" for v in replacements.values())
        consent = st.checkbox("确认删除选中的规则，知晓这可能改变评测口径", key="repair_consent_"+fingerprint) if deleting else True
        if st.button("修复并应用方案", disabled=not consent or "保持待修复" in replacements.values()):
            updated = plan.model_copy(deep=True)
            rules = []
            for i, rule in enumerate(updated.rules):
                action = replacements.get(i)
                if action == "删除此规则":
                    continue
                if action:
                    rule.concept = action
                rules.append(rule)
            updated.rules = rules
            apply_draft(Plan.model_validate(updated.model_dump()))
            st.rerun()


def natural_language_editor(plan, ds, cfg):
    with st.expander("用自然语言修改方案", expanded=True):
        st.caption("基于当前已应用的方案修改；如刚编辑字段或JSON，请先应用对应修改。模型只会生成建议，不会直接替换方案。")
        instruction = st.text_area("告诉模型如何修改", placeholder="例如：删除忠实性维度；将工具结果映射到c24；缺少证据时标记需人工复核。", key="plan_revision_instruction")
        consent = st.checkbox("允许将当前方案、字段样例和修改要求发送到配置的API", key="plan_revision_consent")
        if st.button("生成修改建议", disabled=not (instruction.strip() and consent)):
            st.session_state.pop("revision_candidate", None)
            st.session_state.pop("revision_diagnostic", None)
            try:
                with st.spinner("正在根据要求修改方案…"):
                    updated = revise_plan(plan, instruction, ds.profile() if ds else [], Client(cfg))
                st.session_state["revision_candidate"] = {
                    "base": digest(plan.model_dump()), "plan": updated.model_dump(), "instruction": instruction}
            except CallFailure as error:
                st.error(str(error))
                st.session_state["revision_diagnostic"] = {"summary": str(error), "attempts": error.details}
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error("方案修改失败；原方案未改变")
        candidate = st.session_state.get("revision_candidate")
        if candidate:
            updated = Plan.model_validate(candidate["plan"])
            changed = [key for key, value in plan.model_dump().items() if value != candidate["plan"].get(key)]
            st.write("将修改的部分：", "、".join(changed) or "无变化")
            st.code(updated.model_dump_json(indent=2), language="json")
            st.caption("这只是候选方案，尚未替换当前方案；应用后需要重新确认才能运行。")
            if ds:
                try:
                    for warning in validate_mapping(updated, ds.columns):
                        st.warning(warning)
                except ValueError as error:
                    st.warning(f"候选方案应用后仍需修复：{error}")
            stale = candidate["base"] != digest(plan.model_dump())
            if stale:
                st.warning("当前方案已变化，请重新生成修改建议，避免覆盖新修改。")
            a, b = st.columns(2)
            if a.button("应用修改后的方案", disabled=stale):
                apply_draft(updated)
                st.rerun()
            if b.button("放弃修改建议"):
                st.session_state.pop("revision_candidate", None)
                st.rerun()
        diagnostic = st.session_state.get("revision_diagnostic")
        if diagnostic:
            st.code(json.dumps(diagnostic, ensure_ascii=False, indent=2)[:20000], language="json")
            st.download_button("下载方案修改失败诊断", json.dumps(diagnostic, ensure_ascii=False, indent=2).encode(),
                               "方案修改失败诊断.json", "application/json")
