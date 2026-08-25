"""
GM Advisor — chatbot page for NBA team-building and trade decisions.

Uses a MockLLMClient (no network calls) that mimics the OpenAI
chat.completions interface so a real client can be swapped in later:

    # Real usage would look like:
    # from openai import OpenAI
    # client = OpenAI(api_key=api_key)
    # resp = client.chat.completions.create(model=model, messages=messages)
    # reply = resp.choices[0].message.content
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


# =============================================================================
# MOCK LLM CLIENT
# =============================================================================

# Mock model names — none are wired to a real API yet. Kept here so the UI
# selector is meaningful and a real model list can slot in later.
MOCK_MODELS = [
    "gm-advisor-mock-mini",
    "gm-advisor-mock-standard",
    "gm-advisor-mock-pro",
]


@dataclass
class MockLLMClient:
    """Mimics the OpenAI chat.completions API without any network calls."""

    api_key: str = ""
    model: str = MOCK_MODELS[0]

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        context_bundle: dict[str, Any] | None = None,
    ) -> str:
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        return _mock_generate_reply(user_msg, context_bundle or {}, self.model)


def _mock_generate_reply(user_msg: str, ctx: dict[str, Any], model: str) -> str:
    msg = user_msg.lower()
    intent = _detect_intent(msg)
    parts: list[str] = [_intent_intro(intent, model)]

    if "predictions" in ctx:
        parts.append(_summarize_predictions(ctx["predictions"]))
    if "trade" in ctx:
        parts.append(_summarize_trade(ctx["trade"]))
    if "three_pt" in ctx:
        parts.append(_summarize_three_pt(ctx["three_pt"]))

    if not ctx:
        parts.append(
            "_No page context attached yet — attach outputs from Salary "
            "Predictions, Trade Recommendations, or 3-Point Premium above to "
            "get data-grounded advice._"
        )

    parts.append(_suggest_next_step(intent))
    return "\n\n".join(p for p in parts if p)


def _detect_intent(msg: str) -> str:
    if any(w in msg for w in ("trade", "swap", "deal")):
        return "trade"
    if any(w in msg for w in ("sign", "free agent", "extension", "resign", "re-sign")):
        return "sign"
    if any(w in msg for w in ("cut", "waive", "release", "buyout")):
        return "cut"
    if any(w in msg for w in ("cap", "salary cap", "luxury tax", "apron")):
        return "cap"
    if any(w in msg for w in ("draft", "rookie", "pick")):
        return "draft"
    if any(w in msg for w in ("build", "rebuild", "roster", "core", "long-term", "long term", "plan")):
        return "build"
    return "general"


def _intent_intro(intent: str, model: str) -> str:
    intros = {
        "trade": "Let's evaluate this from a trade-value perspective.",
        "sign": "For a signing / extension decision, here's the read:",
        "cut": "Before waiving anyone, weigh dead cap vs. relief:",
        "cap": "On cap mechanics, keep this in mind:",
        "draft": "For draft strategy, alignment with roster needs matters most:",
        "build": "For roster construction, focus on value alignment first:",
        "general": "Here's my quick read as your GM advisor:",
    }
    return intros[intent]


def _summarize_predictions(pred: dict[str, Any]) -> str:
    lines = ["**From Salary Predictions:**"]
    metrics = pred.get("metrics", {})
    if metrics:
        lines.append(
            f"- Model `{pred.get('model_name', 'n/a')}` — MAE "
            f"${metrics.get('mae', 0):,.0f}, R² {metrics.get('r2', 0):.3f}."
        )
    overpaid = pred.get("overpaid")
    if isinstance(overpaid, pd.DataFrame) and not overpaid.empty:
        r = overpaid.iloc[0]
        lines.append(
            f"- Biggest overpay: **{r['Player']}** — actual "
            f"${r['Actual Salary']:,.0f} vs. predicted "
            f"${r['Predicted Salary']:,.0f} (gap ${r['Difference']:,.0f})."
        )
    underpaid = pred.get("underpaid")
    if isinstance(underpaid, pd.DataFrame) and not underpaid.empty:
        r = underpaid.iloc[0]
        lines.append(
            f"- Best bargain: **{r['Player']}** — worth ~"
            f"${r['Predicted Salary']:,.0f} on a ${r['Actual Salary']:,.0f} deal."
        )
    return "\n".join(lines)


def _summarize_trade(trade: dict[str, Any]) -> str:
    lines = ["**From Trade Recommendations:**"]
    year, team = trade.get("year"), trade.get("team")
    if year and team:
        lines.append(f"- Scope: **{team}**, season **{year}**.")
    counts = trade.get("category_counts", {})
    if counts:
        lines.append(
            f"- League split: {counts.get('High ROI', 0)} High ROI, "
            f"{counts.get('Fair Value', 0)} Fair Value, "
            f"{counts.get('Cap Clog', 0)} Cap Clogs."
        )
    proposed = trade.get("proposed")
    if proposed:
        lines.append(
            f"- Proposed move: send **{proposed['send']}** "
            f"(${proposed['send_salary']:,.0f}); receive "
            f"**{proposed['recv1']}** + **{proposed['recv2']}** — "
            f"cap saved ${proposed['cap_saved']:,.0f}, net production gain "
            f"${proposed['prod_gain']:,.0f}."
        )
    return "\n".join(lines)


def _summarize_three_pt(tp: dict[str, Any]) -> str:
    lines = ["**From 3-Point Premium:**"]
    premium = tp.get("premium_pct")
    if premium is not None:
        direction = "premium" if premium >= 0 else "discount"
        lines.append(
            f"- 3PT specialists earn a **{premium:+.1f}% {direction}** "
            f"vs. efficient interior scorers."
        )
    avg3, avgi = tp.get("avg_3pt"), tp.get("avg_interior")
    if avg3 and avgi:
        lines.append(f"- Avg salary: 3PT ${avg3:,.0f}  |  Interior ${avgi:,.0f}.")
    return "\n".join(lines)


def _suggest_next_step(intent: str) -> str:
    prompts = {
        "trade": "_Want me to sketch a specific two-team framework?_",
        "sign": "_Share the target player and years/$ so I can weigh it against fair value._",
        "cut": "_Tell me the contract details and I'll compare stretch vs. buyout math._",
        "cap": "_Share your current cap sheet or the year you're planning for._",
        "draft": "_What position(s) are you prioritizing?_",
        "build": "_Give me a 1-year or 3-year horizon and I'll shape the plan._",
        "general": "_Ask about a specific team, player, or decision to go deeper._",
    }
    return prompts[intent]


# =============================================================================
# CONTEXT SNAPSHOTS (called from the other pages after they finish computing)
# =============================================================================

def save_predictions_context(
    model_name: str,
    mae: float,
    r2: float,
    top_predicted: pd.DataFrame,
    overpaid: pd.DataFrame,
    underpaid: pd.DataFrame,
) -> None:
    st.session_state["ctx_predictions"] = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "metrics": {"mae": float(mae), "r2": float(r2)},
        "top_predicted": top_predicted.copy(),
        "overpaid": overpaid.copy(),
        "underpaid": underpaid.copy(),
    }


def save_trade_context(
    year: int,
    team: str,
    category_counts: dict[str, int],
    top_roi: pd.DataFrame,
    top_clogs: pd.DataFrame,
    proposed: dict[str, Any] | None,
) -> None:
    st.session_state["ctx_trade"] = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "year": int(year),
        "team": team,
        "category_counts": dict(category_counts),
        "top_roi": top_roi.copy(),
        "top_clogs": top_clogs.copy(),
        "proposed": proposed,
    }


def save_three_pt_context(
    params: dict[str, float],
    avg_3pt: float,
    avg_interior: float,
    premium_pct: float,
    comparison: pd.DataFrame,
    trend: pd.DataFrame | None,
) -> None:
    st.session_state["ctx_three_pt"] = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "avg_3pt": float(avg_3pt),
        "avg_interior": float(avg_interior),
        "premium_pct": float(premium_pct),
        "comparison": comparison.copy(),
        "trend": trend.copy() if isinstance(trend, pd.DataFrame) else None,
    }


# =============================================================================
# BUSINESS PLAN BUILDER
# =============================================================================

def build_business_plan(
    title: str,
    chat_history: list[dict[str, str]],
    ctx_bundle: dict[str, Any],
) -> str:
    """Compose a markdown business plan from the chat + attached page context."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [f"# {title}", f"_Generated {now}_", ""]

    last_user = next(
        (m["content"] for m in reversed(chat_history) if m["role"] == "user"),
        "",
    )
    last_assistant = next(
        (m["content"] for m in reversed(chat_history) if m["role"] == "assistant"),
        "",
    )
    lines += [
        "## Executive Summary",
        f"- **Objective:** {last_user or 'Not specified.'}",
        f"- **Advisor read:** {_first_sentence(last_assistant) or 'No advisor response yet.'}",
        "",
        "## Evidence Used",
    ]
    if not ctx_bundle:
        lines.append("- _No page outputs attached._")
    if "predictions" in ctx_bundle:
        p = ctx_bundle["predictions"]
        lines.append(
            f"- **Salary Predictions** (`{p.get('model_name', 'n/a')}`, "
            f"MAE ${p.get('metrics', {}).get('mae', 0):,.0f}, "
            f"R² {p.get('metrics', {}).get('r2', 0):.3f})."
        )
    if "trade" in ctx_bundle:
        t = ctx_bundle["trade"]
        lines.append(
            f"- **Trade Recommendations** for {t.get('team', 'n/a')} "
            f"({t.get('year', 'n/a')})."
        )
    if "three_pt" in ctx_bundle:
        tp = ctx_bundle["three_pt"]
        lines.append(
            f"- **3-Point Premium**: {tp.get('premium_pct', 0):+.1f}% "
            f"specialist premium."
        )

    lines += ["", "## Key Findings", *_extract_findings(ctx_bundle), ""]
    lines += ["## Recommended Actions", *_extract_actions(ctx_bundle), ""]

    lines.append("## Advisor Conversation")
    for m in chat_history:
        role = "You" if m["role"] == "user" else "Advisor"
        lines.append(f"**{role}:** {m['content']}")
        lines.append("")

    return "\n".join(lines)


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(.+?[.!?])(\s|$)", text.strip())
    return m.group(1) if m else text.strip().splitlines()[0]


def _extract_findings(ctx: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if "predictions" in ctx:
        p = ctx["predictions"]
        over, under = p.get("overpaid"), p.get("underpaid")
        if isinstance(over, pd.DataFrame) and not over.empty:
            r = over.iloc[0]
            out.append(
                f"- Most overpaid: **{r['Player']}** — "
                f"${r['Actual Salary']:,.0f} actual vs. "
                f"${r['Predicted Salary']:,.0f} fair value."
            )
        if isinstance(under, pd.DataFrame) and not under.empty:
            r = under.iloc[0]
            out.append(
                f"- Best value: **{r['Player']}** — worth ~"
                f"${r['Predicted Salary']:,.0f} on a ${r['Actual Salary']:,.0f} deal."
            )
    if "trade" in ctx and ctx["trade"].get("proposed"):
        pr = ctx["trade"]["proposed"]
        out.append(
            f"- Proposed trade frees ${pr['cap_saved']:,.0f} of cap and "
            f"adds ${pr['prod_gain']:,.0f} in production value."
        )
    if "three_pt" in ctx and ctx["three_pt"].get("premium_pct") is not None:
        out.append(
            f"- 3PT specialists earn a {ctx['three_pt']['premium_pct']:+.1f}% "
            f"premium vs. efficient interior scorers."
        )
    if not out:
        out.append("- _Attach page outputs to auto-populate findings._")
    return out


def _extract_actions(ctx: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    n = 1
    if "trade" in ctx and ctx["trade"].get("proposed"):
        pr = ctx["trade"]["proposed"]
        actions.append(
            f"{n}. Explore trading **{pr['send']}** for **{pr['recv1']}** + "
            f"**{pr['recv2']}** to free ${pr['cap_saved']:,.0f} in cap."
        )
        n += 1
    if "predictions" in ctx:
        over = ctx["predictions"].get("overpaid")
        if isinstance(over, pd.DataFrame) and not over.empty:
            r = over.iloc[0]
            actions.append(
                f"{n}. Flag **{r['Player']}** for contract review — "
                f"${abs(r['Difference']):,.0f} above fair value."
            )
            n += 1
    if "three_pt" in ctx:
        prem = ctx["three_pt"].get("premium_pct")
        if prem is not None and prem > 15:
            actions.append(
                f"{n}. Shift acquisition focus toward efficient interior "
                "scorers — market is overpaying 3PT specialists."
            )
            n += 1
    if not actions:
        actions.append(
            "1. _Continue chatting or attach more context to produce concrete actions._"
        )
    return actions


# =============================================================================
# PAGE RENDER
# =============================================================================

def _init_state() -> None:
    st.session_state.setdefault("gm_chat", [])
    st.session_state.setdefault("gm_api_key", "")
    st.session_state.setdefault("gm_model", MOCK_MODELS[0])
    st.session_state.setdefault(
        "gm_ctx_selection",
        {"predictions": False, "trade": False, "three_pt": False},
    )


def _available_context_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    if "ctx_predictions" in st.session_state:
        p = st.session_state["ctx_predictions"]
        labels["predictions"] = (
            f"💰 Salary Predictions — `{p['model_name']}` "
            f"({p['captured_at']})"
        )
    if "ctx_trade" in st.session_state:
        t = st.session_state["ctx_trade"]
        labels["trade"] = (
            f"🎯 Trade Recommendations — {t['team']} / {t['year']} "
            f"({t['captured_at']})"
        )
    if "ctx_three_pt" in st.session_state:
        tp = st.session_state["ctx_three_pt"]
        labels["three_pt"] = (
            f"⭐ 3-Point Premium — {tp['premium_pct']:+.1f}% "
            f"({tp['captured_at']})"
        )
    return labels


def _collect_selected_context() -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    sel = st.session_state["gm_ctx_selection"]
    if sel.get("predictions") and "ctx_predictions" in st.session_state:
        bundle["predictions"] = st.session_state["ctx_predictions"]
    if sel.get("trade") and "ctx_trade" in st.session_state:
        bundle["trade"] = st.session_state["ctx_trade"]
    if sel.get("three_pt") and "ctx_three_pt" in st.session_state:
        bundle["three_pt"] = st.session_state["ctx_three_pt"]
    return bundle


def _preview_context(key: str) -> None:
    if key == "predictions":
        p = st.session_state["ctx_predictions"]
        st.write(
            f"Model `{p['model_name']}` — MAE ${p['metrics']['mae']:,.0f}, "
            f"R² {p['metrics']['r2']:.3f}"
        )
        st.write("Top overpaid:")
        st.dataframe(p["overpaid"].head(3), use_container_width=True)
    elif key == "trade":
        t = st.session_state["ctx_trade"]
        st.write(f"Team: **{t['team']}** — Season {t['year']}")
        st.write(f"Categories: {t['category_counts']}")
        if t.get("proposed"):
            pr = t["proposed"]
            st.write(
                f"Proposed: send **{pr['send']}** → receive "
                f"**{pr['recv1']}** + **{pr['recv2']}** "
                f"(cap saved ${pr['cap_saved']:,.0f})"
            )
    elif key == "three_pt":
        tp = st.session_state["ctx_three_pt"]
        st.write(
            f"3PT avg ${tp['avg_3pt']:,.0f} vs Interior "
            f"${tp['avg_interior']:,.0f} ({tp['premium_pct']:+.1f}%)"
        )


def render_gm_advisor_page() -> None:
    _init_state()

    st.title("🤖 GM Advisor")
    st.caption(
        "A chat-based assistant for team-building and trade decisions. "
        "Attach outputs from other pages to ground the advice in your data."
    )

    # --- Sidebar: advisor settings ---
    st.sidebar.header("🔑 Advisor Settings")
    st.session_state["gm_api_key"] = st.sidebar.text_input(
        "OpenAI API key",
        value=st.session_state["gm_api_key"],
        type="password",
        help="Stored only in this session. The mock backend ignores it.",
    )
    st.session_state["gm_model"] = st.sidebar.selectbox(
        "Model",
        options=MOCK_MODELS,
        index=MOCK_MODELS.index(st.session_state["gm_model"]),
        help="Mock models only — real OpenAI models are not wired in yet.",
    )
    st.sidebar.caption(
        "_Backend is currently a local mock — no network calls are made._"
    )
    if st.sidebar.button("🧹 Clear conversation", use_container_width=True):
        st.session_state["gm_chat"] = []
        st.session_state.pop("gm_plan_md", None)
        st.rerun()

    # --- Context attach panel ---
    st.subheader("📎 Attach Page Context")
    labels = _available_context_labels()
    if not labels:
        st.info(
            "Run any of the other pages (Salary Predictions, Trade "
            "Recommendations, 3-Point Premium) once and their outputs will "
            "appear here for you to attach to the conversation."
        )
    else:
        cols = st.columns(len(labels))
        for i, (key, label) in enumerate(labels.items()):
            with cols[i]:
                st.session_state["gm_ctx_selection"][key] = st.checkbox(
                    label,
                    value=st.session_state["gm_ctx_selection"].get(key, False),
                    key=f"gm_ctx_cb_{key}",
                )
                if st.session_state["gm_ctx_selection"][key]:
                    with st.expander("Preview attached data", expanded=False):
                        _preview_context(key)

    st.markdown("---")

    # --- Conversation ---
    st.subheader("💬 Conversation")
    for msg in st.session_state["gm_chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(
        "Ask about a trade, signing, cap plan, or roster build..."
    )
    if user_prompt:
        st.session_state["gm_chat"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        client = MockLLMClient(
            api_key=st.session_state["gm_api_key"],
            model=st.session_state["gm_model"],
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = client.chat_completion(
                    messages=st.session_state["gm_chat"],
                    context_bundle=_collect_selected_context(),
                )
            st.markdown(reply)
        st.session_state["gm_chat"].append({"role": "assistant", "content": reply})

    # --- Business plan export ---
    st.markdown("---")
    st.subheader("📄 Business Plan")

    if not st.session_state["gm_chat"]:
        st.caption("Start a conversation to enable business plan generation.")
        return

    col_a, col_b = st.columns([2, 1])
    with col_a:
        plan_title = st.text_input(
            "Plan title", value="NBA Team-Building Business Plan"
        )
    with col_b:
        generate = st.button(
            "🧾 Generate Business Plan",
            type="primary",
            use_container_width=True,
        )

    if generate:
        st.session_state["gm_plan_md"] = build_business_plan(
            title=plan_title,
            chat_history=st.session_state["gm_chat"],
            ctx_bundle=_collect_selected_context(),
        )

    if "gm_plan_md" in st.session_state:
        with st.expander("📖 Preview business plan", expanded=True):
            st.markdown(st.session_state["gm_plan_md"])
        st.download_button(
            label="⬇️ Download plan (Markdown)",
            data=st.session_state["gm_plan_md"].encode("utf-8"),
            file_name=f"business_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
