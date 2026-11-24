from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from judgekit.engine import Registry
from judgekit.connection import probe_connection
from judgekit.files import display, export_dataset, load_dataset, workbook_sheets
from judgekit.llm import Client, CallFailure, redact
from judgekit.models import APIConfig, Plan, output_columns
from judgekit.plans import propose_plan, remap_plan, validate_mapping
from judgekit.plan_editor import edit_fields, natural_language_editor, repair_references
from judgekit.references import parse_reference
from judgekit.storage import Store, digest

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="裁判工作台 · 批量评测", page_icon="◎", layout="wide")
st.markdown("""<style>
.block-container{max-width:1400px;padding-top:2rem}
h1{font-size:2rem!important;letter-spacing:-.04em} h2{font-size:1.3rem!important}
[data-testid="stMetric"]{background:#f1f5f8;border-radius:12px;padding:12px}
.stButton button{border-radius:8px} footer{visibility:hidden}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def registry():
    return Registry()


store = Store(Path(os.environ.get("JUDGE_DATA_DIR", str(ROOT / "data"))) / "jobs")


def show_plan_diagnostic(payload):
    if not payload:
        return
    st.markdown("#### 最近一次方案生成失败详情")
    st.caption("诊断只保留在当前页面会话中，不写入任务结果或本地日志；下载内容可能包含模型复述的裁判规则或字段样例。")
    for item in payload.get("attempts", []):
        label = f"第 {item.get('attempt', '?')} 次：{item.get('reason', '未知原因')}"
        with st.expander(label, expanded=True):
            facts = []
            if item.get("http_status") is not None:
                facts.append(f"HTTP {item['http_status']}")
            if item.get("finish_reason"):
                facts.append(f"finish_reason={item['finish_reason']}")
            if facts:
                st.caption(" · ".join(facts))
            raw = item.get("assistant_content") or item.get("response_body") or item.get("refusal")
            if raw:
                shown = raw[:20_000]
                st.code(shown, language="json")
                if len(raw) > len(shown) or item.get("content_truncated"):
                    st.caption("页面仅展示前20,000字符；下载文件中包含工具保留下来的更完整内容。")
            else:
                st.caption("本次没有收到可展示的模型正文。")
    diagnostic_json = json.dumps(payload, ensure_ascii=False, indent=2)
    left, right = st.columns([1, 1])
    left.download_button("下载失败诊断（JSON）", diagnostic_json.encode("utf-8"),
                         file_name="方案生成失败诊断.json", mime="application/json",
                         width="stretch")
    if right.button("清除失败诊断", width="stretch"):
        st.session_state.pop("last_plan_diagnostic", None)
        st.rerun()


def get_config():
    with st.sidebar:
        st.markdown("### 模型连接")
        mock = st.toggle("演示模式（无需Key）", value=True,
                         help="只验证流程，所有结论标记为模拟，不评判实际质量。")
        provider = st.text_input("API提供方（备注）", value="OpenAI兼容")
        base = st.text_input("API Base URL", value="https://api.openai.com/v1")
        model = st.text_input("裁判模型名称", placeholder="填写服务提供方的模型ID")
        key = st.text_input("API Key", type="password", help="仅保留在当前进程内存；重启后重新输入。")
        mode = st.selectbox("结构化输出", ["json_object", "json_schema", "plain"],
                            help="json_schema更严格，但服务商可能不支持；不自动静默降级。")
        with st.expander("高级配置"):
            temperature_on = st.checkbox("发送Temperature", value=True)
            temperature = st.number_input("Temperature", 0.0, 2.0, 0.0, 0.1)
            timeout = st.number_input("单次超时（秒）", 1, 600, 60)
            concurrency = st.number_input("最大并发", 1, 16, 2)
            retries = st.number_input("额外重试次数", 0, 5, 2)
            rpm = st.number_input("每分钟请求上限", 1, 6000, 60)
            max_chars = st.number_input("单条最大输入字符数", 1000, 500000, 60000, 1000)
            output_tokens = st.number_input("最大输出Token", 256, 32000, 4000)
            token_parameter = st.selectbox("输出长度参数名", ["max_completion_tokens", "max_tokens"], help="部分兼容服务只支持max_tokens；请按服务商文档选择。")
            price_in = st.number_input("每百万输入Token价格（自填）", min_value=0.0, value=0.0)
            price_out = st.number_input("每百万输出Token价格（相同币种）", min_value=0.0, value=0.0)
        cfg = APIConfig(provider=provider, base_url=base, model=model, api_key=key, mode=mode,
                        temperature=temperature if temperature_on else None, timeout=timeout,
                        concurrency=concurrency, retries=retries, rpm=rpm, max_chars=max_chars,
                        max_output_tokens=output_tokens, token_parameter=token_parameter, input_price=price_in, output_price=price_out, mock=mock)
        st.caption("连接测试独立使用：20秒总等待上限 · 不重试 · 最大输出512 Token")
        if st.button("测试连接", use_container_width=True):
            if cfg.mock:
                st.success("演示模式可用；未连接真实模型")
            else:
                with st.status("正在测试连接…", expanded=True) as connection_status:
                    def show_progress(elapsed, message):
                        connection_status.write(f"{elapsed:.1f}秒 · {message}")
                    report = probe_connection(cfg, show_progress)
                    if report.status == "ok":
                        connection_status.update(label=f"测试通过 · {report.elapsed:.1f}秒", state="complete")
                    elif report.status in {"output_incomplete", "format_error", "invalid_response", "refused"}:
                        connection_status.update(label=f"已有响应，格式待确认 · {report.elapsed:.1f}秒", state="error")
                    else:
                        connection_status.update(label=f"测试未通过 · {report.elapsed:.1f}秒", state="error")
                if report.status == "ok":
                    st.success(f"当前模式测试通过 · {report.elapsed:.1f}秒")
                elif report.status in {"output_incomplete", "format_error", "invalid_response", "refused", "rate_limited"}:
                    st.warning(report.message)
                else:
                    st.error(report.message)
                st.caption("仅验证当前模型和输出模式；不读取提测文件或参考资料，不改变正式评测的超时、重试或输出上限。")
        st.divider()
        st.caption("本地存储不等于数据不出网：分析字段和评判时，所选内容会发送到你配置的API。运行目录含原始文件、证据和结果，请妥善保管。")
        return cfg


cfg = get_config()
st.title("裁判工作台")
st.caption("任意表头 · 自定义规则 · 参考资料 · 可追溯的批量评测")
if cfg.mock:
    st.info("当前为演示模式：不消耗API费用，不产生真实评分或通过结论。切换侧栏后可使用自己的模型。")

upload_tab, plan_tab, confirm_tab, run_tab = st.tabs(["① 上传与参考资料", "② 生成评测方案", "③ 确认方案", "④ 运行与导出"])

with upload_tab:
    st.subheader("提测数据")
    demo = st.checkbox("加载内置演示数据", key="demo_data")
    uploaded = st.file_uploader("上传提测文件", type=["xlsx", "csv", "json", "jsonl"])
    source = (ROOT/"examples"/"cases.csv").read_bytes() if demo else (uploaded.getvalue() if uploaded else None)
    name = "cases.csv" if demo else (uploaded.name if uploaded else "")
    ds = None
    if source:
        try:
            sheet = st.selectbox("待评测工作表", workbook_sheets(source)) if name.lower().endswith(".xlsx") else ""
            header = st.number_input("表头行号", 1, 100, 1) if name.lower().endswith((".xlsx", ".csv")) else 1
            json_root = st.text_input("对象数组所在顶层键（数组文件留空）") if name.lower().endswith(".json") else ""
            ds = load_dataset(source, name, sheet, header, json_root)
            st.success(f"已读取 {len(ds.rows)} 条数据 · {len(ds.labels)} 列")
            st.caption("第一版按一行一个用例。历史对话需存放在该行的某列；不会擅自把上一行拼成历史。Excel公式不会重新计算。")
            preview = [{f"{k} · {ds.columns[k]}": display(v)[:800] for k, v in r.items()} for r in ds.rows[:8]]
            st.dataframe(preview, use_container_width=True, hide_index=True)
            st.session_state["dataset"] = ds
        except Exception as e:
            st.error(str(e) if isinstance(e, ValueError) else "文件读取失败，请检查格式、加密状态和表头")
    # Do not silently keep an older file when current upload fails or is cleared.
    if ds is None:
        st.session_state.pop("dataset", None)

    st.divider()
    st.subheader("参考资料（可选，可多份）")
    refs = st.file_uploader("知识库、标准答案、工具说明或业务规范", type=["txt", "md", "pdf", "docx", "xlsx", "csv", "json", "jsonl"], accept_multiple_files=True)
    include_demo_ref = st.checkbox("加入演示知识库", value=demo)
    items = [(u.name, u.getvalue()) for u in refs]
    if include_demo_ref:
        items.append(("knowledge.md", (ROOT/"examples"/"knowledge.md").read_bytes()))
    documents = []
    for num, (ref_name, data) in enumerate(items):
        ref_key = digest([ref_name, len(data), data.hex()[:64], __import__("hashlib").sha256(data).hexdigest()])
        with st.expander(f"{ref_name} · {len(data)/1024:.1f} KB", expanded=True):
            if st.checkbox("本次使用", value=True, key="use_"+ref_key):
                try:
                    cache_key = "parsed_"+ref_key
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = [asdict(c) for c in parse_reference(data, ref_name)]
                    chunks = st.session_state[cache_key]
                    st.caption(f"解析成功 · {len(chunks)}个片段 · {sum(len(c['text']) for c in chunks)}字")
                    st.text(chunks[0]["text"][:500])
                    mode_name = st.selectbox("使用方式", ["自动检索", "全局参考", "按字段匹配"], key="mode_"+ref_key)
                    required = st.checkbox("此资料为必需证据（无匹配时转复核）", key="required_"+ref_key)
                    doc = {"name": ref_name, "chunks": chunks, "required": required,
                           "mode": {"自动检索": "auto", "全局参考": "global", "按字段匹配": "match"}[mode_name]}
                    if doc["mode"] == "match":
                        if ds:
                            doc["column_id"] = st.selectbox("提测文件匹配字段", list(ds.columns), format_func=lambda k: f"{k} · {ds.columns[k]}", key="match_"+ref_key)
                        fields = list(dict.fromkeys(k for c in chunks for k in c.get("fields", {})))
                        if not fields:
                            st.warning("字段匹配需要结构化资料，例如CSV、Excel或对象数组JSON；此文件只能全局或自动检索。")
                            continue
                        doc["reference_field"] = st.selectbox("参考资料标识字段", fields, key="refmatch_"+ref_key)
                    documents.append(doc)
                except Exception as e:
                    st.error(str(e) if isinstance(e, ValueError) else "资料解析失败，请检查文件格式")
    st.session_state["documents"] = documents
    st.caption("可用上传组件的 × 删除文件，也可取消‘本次使用’。扫描PDF不支持；PDF、Word中的图片不做识别。")

with plan_tab:
    st.subheader("告诉裁判：这次怎样评")
    default_rule = (ROOT/"examples"/"rules.md").read_text(encoding="utf-8") if demo else ""
    rule_upload = st.file_uploader("或上传裁判规则", type=["txt", "md", "docx"])
    text = st.text_area("裁判规则", value=default_rule, height=230, help="说明评测对象、评分档位、硬性禁区和通过条件。没有定义的标准不会擅自补齐。")
    if rule_upload:
        try:
            text += "\n"+"\n".join(c.text for c in parse_reference(rule_upload.getvalue(), rule_upload.name))
        except Exception:
            st.error("规则文件读取失败")
    consent = st.checkbox("允许将字段样例和裁判规则发送到我配置的API（演示模式不发送）")
    if st.button("理解规则并推荐字段", type="primary", disabled=not (ds and text.strip() and (consent or cfg.mock))):
        st.session_state.pop("last_plan_diagnostic", None)
        try:
            with st.spinner("正在整理评测方案…"):
                p = propose_plan(text, ds.profile(), Client(cfg))
            st.session_state["plan_draft"] = p.model_dump_json(indent=2)
            st.session_state["plan_fingerprint"] = ds.fingerprint
            st.session_state.pop("confirmed", None)
            st.success("已生成，请到‘③ 确认方案’检查；自动映射置信度只是提示，不是可靠概率。")
        except CallFailure as e:
            st.error(str(e))
            if e.details:
                st.session_state["last_plan_diagnostic"] = {
                    "summary": str(e), "error_type": e.kind, "model": cfg.model,
                    "output_mode": cfg.mode, "attempts": e.details,
                }
        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error("方案生成失败，请检查API配置")
    show_plan_diagnostic(st.session_state.get("last_plan_diagnostic"))
    st.divider()
    plan_upload = st.file_uploader("导入之前保存的评测方案（JSON）", type=["json"])
    if plan_upload and st.button("加载方案"):
        try:
            p = Plan.model_validate_json(plan_upload.getvalue())
            st.session_state["plan_draft"] = p.model_dump_json(indent=2)
            st.session_state.pop("confirmed", None)
            st.success("方案已加载；若列名变化，请重新映射")
        except Exception:
            st.error("不是有效的评测方案JSON")
    if ds and st.session_state.get("plan_draft") and st.button("保留规则，仅为当前文件重新匹配字段", disabled=not (consent or cfg.mock)):
        try:
            p = remap_plan(Plan.model_validate_json(st.session_state["plan_draft"]), ds.profile(), Client(cfg))
            st.session_state["plan_draft"] = p.model_dump_json(indent=2)
            st.session_state.pop("confirmed", None)
            st.success("已重新推荐字段，请检查确认")
        except Exception:
            st.error("重新匹配失败，请检查API配置或手动编辑方案")

with confirm_tab:
    st.subheader("先确认一次，再批量评")
    plan = None
    if st.session_state.get("plan_draft"):
        try:
            plan = Plan.model_validate_json(st.session_state["plan_draft"])
            st.markdown(f"**{plan.name}** · v{plan.version}")
            st.write("评测对象：", plan.evaluation_target)
            st.write("通过条件：", plan.pass_condition)
            st.write("重大错误：", plan.hard_fail_rules or "未单独定义")
            st.dataframe([{"概念": c.description, "来源": c.source, "字段": ", ".join(c.column_ids),
                           "置信提示": c.confidence, "说明": c.explanation} for c in plan.concepts], hide_index=True)
            natural_language_editor(plan, ds, cfg)
            edit_fields(plan, ds)
            repair_references(plan)
            st.markdown("**评分维度**")
            st.dataframe([{"名称": d.name, "合法值": display(d.allowed_values), "标准": d.rubric, "说明": d.description} for d in plan.dimensions], hide_index=True)
            with st.expander("将生成的结果列"):
                st.dataframe({"结果列": output_columns(plan)}, hide_index=True)
            with st.expander("完整方案（可修改维度、规则、通过条件）"):
                edited = st.text_area("方案JSON", value=st.session_state["plan_draft"], height=380, key="json_"+digest(st.session_state["plan_draft"]))
                if st.button("应用方案JSON"):
                    try:
                        updated = Plan.model_validate_json(edited)
                        st.session_state["plan_draft"] = updated.model_dump_json(indent=2)
                        st.session_state.pop("confirmed", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"方案结构不符合要求：{str(e)[:500]}")
            st.download_button("保存评测方案", plan.model_dump_json(indent=2), "评测方案.json", "application/json")
            if ds:
                mapping_valid = True
                try:
                    warnings = validate_mapping(plan, ds.columns)
                except ValueError as error:
                    mapping_valid, warnings = False, []
                    st.error(str(error)+"；可在上方修复规则引用、调整字段或用自然语言修改方案。")
                for warning in warnings:
                    st.warning(warning+"；运行时会标记证据不足，不会脑补")
                if st.button("确认此方案", type="primary", disabled=not mapping_valid):
                    st.session_state["confirmed"] = digest([plan.model_dump(), ds.fingerprint])
                    st.success("方案已确认，进入‘④ 运行与导出’")
        except Exception as e:
            st.error(str(e)[:500])
    else:
        st.caption("请先生成或导入评测方案")

with run_tab:
    st.subheader("开始一个评测批次")
    plan = Plan.model_validate_json(st.session_state["plan_draft"]) if st.session_state.get("plan_draft") else None
    if ds and plan:
        confirmed = st.session_state.get("confirmed") == digest([plan.model_dump(), ds.fingerprint])
        if not confirmed:
            st.warning("文件或方案尚未确认。请先完成③；修改后需重新确认。")
        first, last = st.columns(2)
        begin = first.number_input("起始数据行（第1条起）", 1, len(ds.rows), 1)
        end = last.number_input("结束数据行", begin, len(ds.rows), len(ds.rows))
        selection = st.text_input("或指定数据行编号（例如 1,3,7；留空用上述范围）")
        answer_columns = [k for k, label in ds.columns.items() if any(term in label.lower() for term in ["待评回复", "最终回复", "response", "answer", "模型输出"])]
        default_anchor = answer_columns[0] if answer_columns else list(ds.columns)[-1]
        anchor = st.selectbox("把结果插在此列右侧", list(ds.columns), index=list(ds.columns).index(default_anchor), format_func=lambda k: f"{k} · {ds.columns[k]}")
        st.caption("Excel公式/合并/复杂对象若不适合中间插列，导出时会要求改为最右侧；不会覆盖原文件。")
        default_queries = [k for k, label in ds.columns.items() if any(term in label.lower() for term in ["用户问题", "query", "用户输入", "request", "prompt"])]
        ref_query_cols = st.multiselect("参考资料检索使用哪些输入列", list(ds.columns), default=default_queries,
                                        format_func=lambda k: f"{k} · {ds.columns[k]}",
                                        help="建议只选用户问题和必要上下文，不选被评回答，防止幻觉引导检索。")
        x, y = st.columns(2)
        top_k = x.number_input("每份资料最多检索片段", 1, 30, 5)
        ref_chars = y.number_input("参考资料总字符预算", 1000, 100000, 10000, 1000)
        existing_mode = st.selectbox("若原文件已有相同评测列", ["新版本列（保留旧结果）", "跳过已有结果的行", "覆盖已有judge_列"])
        default_suffix = "_v2" if existing_mode == "新版本列（保留旧结果）" and any(n.startswith("judge_") for n in ds.labels) else ""
        suffix = st.text_input("结果列版本后缀（可留空）", value=default_suffix)
        if suffix and not re.fullmatch(r"[\w.-]{1,40}", suffix):
            st.warning("后缀只允许字母、数字、中文、下划线、点或短横线")
        consent_run = st.checkbox("确认：所选证据可发送给配置的API；允许将提测文件、方案、资料和结果保存在本机，以便恢复任务")
        if st.button("创建并开始评测", type="primary", disabled=not (confirmed and consent_run)):
            try:
                selected = sorted(set(int(s.strip())-1 for s in selection.split(",") if s.strip())) if selection else list(range(begin-1, end))
                if not selected or any(i < 0 or i >= len(ds.rows) for i in selected):
                    raise ValueError("选择的数据行超出范围")
                if suffix and not re.fullmatch(r"[\w.-]{1,40}", suffix):
                    raise ValueError("结果列后缀格式不正确")
                if existing_mode == "新版本列（保留旧结果）" and any(k+suffix in ds.labels for k in output_columns(plan)):
                    raise ValueError("本版本结果列已存在，请使用新的后缀，或选择跳过/覆盖已有评测列")
                if any(d["mode"] == "auto" for d in documents) and not ref_query_cols:
                    raise ValueError("自动检索需要指定至少一个检索输入列")
                if not cfg.mock and (not cfg.api_key or not cfg.model):
                    raise ValueError("请先填写模型名称和API Key")
                if cfg.api_key and cfg.api_key in json.dumps(ds.rows, ensure_ascii=False, default=str):
                    raise ValueError("上传的数据中包含当前API Key；请移除后再运行，以免写入原文件副本")
                if existing_mode == "跳过已有结果的行":
                    status_id = next((k for k, v in ds.columns.items() if v == "judge_status"+suffix), None)
                    if status_id:
                        selected = [i for i in selected if not ds.rows[i].get(status_id)]
                if not selected:
                    raise ValueError("没有剩余的待评数据")
                settings = {"api": cfg.model_dump(), "top_k": top_k, "reference_chars": ref_chars,
                            "retrieval_columns": ref_query_cols, "suffix": suffix,
                            "collision": "overwrite" if existing_mode != "新版本列（保留旧结果）" else "new"}
                safe_plan = Plan.model_validate(redact(plan.model_dump(), cfg.api_key))
                safe_documents = redact(documents, cfg.api_key)
                job_id = store.create(ds, safe_plan, safe_documents, redact(settings, cfg.api_key), selected, anchor)
                registry().launch(store, job_id, cfg)
                st.session_state["active_job"] = job_id
                st.success("已启动；可以在下方查看进度")
            except Exception as e:
                st.error(str(e) if isinstance(e, ValueError) else "启动失败，请检查配置")
    st.divider()
    st.subheader("运行记录与断点恢复")
    jobs = store.list()
    if jobs:
        ids = [j["id"] for j in jobs]
        chosen = st.session_state.get("active_job", ids[0])
        job_id = st.selectbox("选择运行记录", ids, index=ids.index(chosen) if chosen in ids else 0,
                              format_func=lambda x: next(j["filename"]+" · "+j["name"]+" · "+x[:8] for j in jobs if j["id"] == x))

        @st.fragment(run_every="2s")
        def job_panel(job_id):
            meta = store.meta(job_id)
            results = store.results(job_id)
            worker = registry().workers.get(job_id)
            alive = bool(worker and worker.thread.is_alive())
            status = worker.state if worker else "可从本地记录恢复"
            st.caption(f"状态：{status} · 编号：{job_id}")
            total = len(meta["selected"])
            st.progress(min(len(results)/max(total, 1), 1), text=f"已完成 {len(results)} / {total}")
            counts = [p["flat"] for p in results.values()]
            a, b, c, d = st.columns(4)
            a.metric("待人工复核", sum(bool(p.get("judge_needs_review")) for p in counts))
            b.metric("请求/格式失败", sum(p.get("judge_status") in {"api_error", "format_error", "internal_error", "refused"} for p in counts))
            c.metric("Token（已记录）", sum(p.get("judge_input_tokens", 0)+p.get("judge_output_tokens", 0) for p in counts))
            costs = [p.get("judge_cost") for p in counts if p.get("judge_cost") is not None]
            d.metric("估算费用（自填币种）", f"{sum(costs):.4f}" if costs else "未配置价格")
            seconds = sum(p.get("seconds", 0) for p in results.values())
            st.caption(f"累计用例处理时间 {seconds:.1f}s（并发时非墙钟时间）；创建时间 {time.strftime('%Y-%m-%d %H:%M', time.localtime(meta['created']))}。格式重试计入Token；网络超时的远端计费可能无法获知。")
            if alive:
                a, b, c = st.columns(3)
                if a.button("暂停新请求", key="pause_"+job_id):
                    worker.pause()
                if b.button("继续", key="resume_"+job_id):
                    worker.resume()
                if c.button("停止", key="stop_"+job_id):
                    worker.stop()
                st.caption("暂停/停止不撤销已发出的请求；正在执行的请求会完成或超时并保存。")
            else:
                retry_mode = st.selectbox("恢复范围", ["pending", "failed", "review"], format_func=lambda m: {"pending": "仅未完成的行", "failed": "未完成及请求失败的行", "review": "未完成及需复核的行"}[m], key="retry_"+job_id)
                if st.button("恢复本次运行", key="restart_"+job_id):
                    try:
                        saved_cfg = APIConfig.model_validate(meta["settings"]["api"])
                        saved_cfg.api_key = cfg.api_key
                        registry().launch(store, job_id, saved_cfg, retry_mode)
                        st.success("恢复中；使用该批次原模型、方案和资料，仅Key取自当前输入")
                    except Exception as e:
                        st.error(str(e) if isinstance(e, ValueError) else "恢复失败")
            if worker and worker.error:
                st.error(worker.error)
            if not results:
                return
            filter_name = st.selectbox("筛选结果", ["全部", "通过", "不通过", "需要人工复核", "失败/证据不足"], key="filter_"+job_id)
            view = {}
            for i, p in results.items():
                f = p["flat"]
                if filter_name == "通过" and f.get("judge_overall_pass") is not True:
                    continue
                if filter_name == "不通过" and f.get("judge_overall_pass") is not False:
                    continue
                if filter_name == "需要人工复核" and not f.get("judge_needs_review"):
                    continue
                if filter_name == "失败/证据不足" and f.get("judge_status") == "completed":
                    continue
                view[i] = f
            # Full JSON stays in detail view, not every table cell.
            preview = [{"数据行": i+1, **{k: display(v) for k, v in f.items() if k != "judge_raw_json"}} for i, f in view.items()]
            st.dataframe(preview, use_container_width=True, hide_index=True)
            if view:
                error_filter = st.text_input("错误类型包含（可留空）", key="err_filter_"+job_id)
                dimension_cols = [k for f in view.values() for k in f if k.startswith("judge_") and isinstance(f[k], (int, float)) and not isinstance(f[k], bool)]
                with st.expander("按分数筛选"):
                    score_col = st.selectbox("数字结果列", ["不筛选", *dict.fromkeys(dimension_cols)], key="scorecol_"+job_id)
                    score_max = st.number_input("只看不高于此分数", value=3.0, key="scoremax_"+job_id)
                if error_filter:
                    view = {i: f for i, f in view.items() if error_filter in display(f.get("judge_error_type"))}
                if score_col != "不筛选":
                    view = {i: f for i, f in view.items() if isinstance(f.get(score_col), (int, float)) and f[score_col] <= score_max}
                st.caption(f"细筛后 {len(view)} 条")
                if error_filter or score_col != "不筛选":
                    st.dataframe([{"数据行": i+1, **{k: display(v) for k, v in f.items() if k != "judge_raw_json"}} for i, f in view.items()], hide_index=True)
            if view:
                index = st.selectbox("查看单条完整证据", list(view), format_func=lambda i: f"第{i+1}条", key="detail_"+job_id)
                with st.expander("裁判输入、原始输出与程序检查"):
                    st.json(results[index])
            st.download_button("下载运行详情JSON（含证据，请妥善保管）", json.dumps(results, ensure_ascii=False, indent=2), "评测运行详情.json", "application/json", key="log_"+job_id)
            data = (store.path(job_id)/"source.bin").read_bytes()
            dataset = load_dataset(data, meta["filename"], meta["sheet"], meta["header_row"], meta["json_root"])
            export_anchor = st.selectbox("导出插入位置", list(dataset.columns), index=list(dataset.columns).index(meta["anchor"]), format_func=lambda k: f"{k} · {dataset.columns[k]}", key="exportanchor_"+job_id)
            export_suffix = st.text_input("导出列后缀", value=meta["settings"].get("suffix", ""), key="suffix_"+job_id)
            export_key = digest([job_id, len(results), export_anchor, export_suffix, results])
            if st.button("生成结果文件", key="export_"+job_id):
                try:
                    flat = {i: p["flat"] for i, p in results.items()}
                    content, filename = export_dataset(dataset, flat, export_anchor, export_suffix, collision=meta["settings"].get("collision", "new"))
                    main_export = (content, filename)
                    bad = [i for i, f in flat.items() if f.get("judge_needs_review") or f.get("judge_overall_pass") is False]
                    content, filename = export_dataset(dataset, flat, export_anchor, export_suffix, only_rows=bad, collision="overwrite")
                    st.session_state["export_files_"+job_id] = (export_key, main_export, (content, filename))
                except ValueError as e:
                    st.error(str(e))
            saved_export = st.session_state.get("export_files_"+job_id)
            if saved_export and saved_export[0] == export_key:
                _, main_export, bad_export = saved_export
                st.download_button("下载完整结果文件", main_export[0], main_export[1], key="download_"+job_id)
                st.download_button("下载待复核与异常用例", bad_export[0], bad_export[1], "text/csv", key="bad_"+job_id)

        job_panel(job_id)
    else:
        st.caption("尚无运行记录。完成前三步后开始；也可以先加载演示数据体验。")
