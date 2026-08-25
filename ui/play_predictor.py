"""
Play Recognition — in-the-works page.

Three jobs:
    1. Train a vision model on labeled play images (coming soon).
    2. Review / edit the canonical play list stored in data/plays.md.
    3. Predict a play from 1-5 uploaded frames (coming soon).

Only the plays-catalog editing job is functional today. The training and
prediction backends are stubbed so the UI is ready to plug into a future
vision classifier.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PLAYS_FILE = Path("data/plays.md")
_HEADER_DEFAULT = "Basketball 1.0"

# Matches en/em dash, spaced hyphen, or ": " between name and description.
_SPLIT_RE = re.compile(r"\s+[–—-]\s+|:\s+")


# =============================================================================
# PLAYS FILE I/O
# =============================================================================

def load_plays(path: Path = PLAYS_FILE) -> tuple[str, pd.DataFrame]:
    """Return (header, DataFrame[name, description]) parsed from the plays file."""
    if not path.exists():
        return _HEADER_DEFAULT, pd.DataFrame(columns=["name", "description"])

    lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8").splitlines()]
    header = _HEADER_DEFAULT
    rows: list[dict[str, str]] = []
    header_captured = False

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("•"):
            body = stripped.lstrip("•").strip()
            match = _SPLIT_RE.search(body)
            if match:
                name = body[: match.start()].strip()
                desc = body[match.end() :].strip()
            else:
                name, desc = body, ""
            rows.append({"name": name, "description": desc})
        elif not header_captured:
            header = stripped
            header_captured = True

    return header, pd.DataFrame(rows, columns=["name", "description"])


def save_plays(header: str, df: pd.DataFrame, path: Path = PLAYS_FILE) -> int:
    """Write the plays back to `path` in the original bullet format. Returns count."""
    lines: list[str] = [header, ""]
    written = 0
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        desc = str(row.get("description", "")).strip()
        lines.append(f"•\t{name} – {desc}" if desc else f"•\t{name}")
        written += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


# =============================================================================
# MODEL STUBS (to be replaced when the vision model ships)
# =============================================================================

def predict_play_stub(images: list[Any], play_names: list[str]) -> dict[str, Any]:
    return {
        "status": "coming_soon",
        "message": (
            "The play-recognition model is not yet trained or wired in. "
            "This UI is ready to plug into a future vision classifier."
        ),
        "candidates": [],
        "image_count": len(images),
        "vocab_size": len(play_names),
    }


def train_model_stub(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "coming_soon",
        "message": (
            "Training pipeline is not yet implemented. Uploaded samples were "
            "counted but not persisted."
        ),
        "sample_count": len(samples),
    }


# =============================================================================
# PAGE RENDER
# =============================================================================

def render_play_predictor_page() -> None:
    st.title("🎬 Play Recognition")
    st.markdown(
        '<span style="background:#fff3cd;color:#664d03;padding:2px 10px;'
        'border-radius:12px;font-size:0.85em;font-weight:600;">'
        '🚧 In the works</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Train a computer-vision model to recognize NBA half-court plays from "
        "1-5 still frames. Review and curate the canonical play list, then run "
        "predictions on new uploads."
    )
    st.info(
        "**Concept:** given a handful of stills from a possession, classify "
        "the play (e.g. `Horns`, `Spain P&R`, `Zoom`) using a fixed vocabulary "
        "defined in `data/plays.md`. The UI is ready; the vision model is not "
        "yet trained."
    )

    header, plays_df = load_plays()
    st.session_state.setdefault("plays_header", header)
    st.session_state.setdefault("plays_df", plays_df.copy())

    tab_train, tab_review, tab_predict = st.tabs(
        ["🏋️ 1. Train Model", "📖 2. Review Plays", "🔮 3. Predict Play"]
    )

    with tab_train:
        _render_train_tab()
    with tab_review:
        _render_review_tab()
    with tab_predict:
        _render_predict_tab()


def _render_train_tab() -> None:
    st.subheader("Train the play-recognition model")
    st.warning("🚧 **Coming soon** — training pipeline is not yet implemented.")
    st.markdown(
        "Upload labeled sample frames per play so we can eventually fine-tune "
        "a small vision model on the fixed play vocabulary."
    )

    plays_df: pd.DataFrame = st.session_state["plays_df"]
    play_options = plays_df["name"].dropna().tolist() or ["(no plays defined)"]

    col1, col2 = st.columns([1, 2])
    with col1:
        label = st.selectbox("Play label", options=play_options, key="train_label")
    with col2:
        uploads = st.file_uploader(
            "Sample frames (1-5 per possession)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="train_uploads",
        )

    if uploads:
        st.caption(f"Selected {len(uploads)} file(s) for label **{label}**.")
        cols = st.columns(min(len(uploads), 5))
        for i, f in enumerate(uploads[:5]):
            with cols[i]:
                st.image(f, use_container_width=True)

    st.button(
        "Submit training samples",
        disabled=True,
        use_container_width=True,
        help="Will activate once the training backend ships.",
    )


def _render_review_tab() -> None:
    st.subheader("Review & edit the play catalog")
    st.markdown(
        f"The predictor uses a fixed vocabulary of plays. This list is stored "
        f"in `{PLAYS_FILE.as_posix()}` — edit inline, add rows, then save."
    )

    st.session_state["plays_header"] = st.text_input(
        "Catalog title",
        value=st.session_state["plays_header"],
    )

    edited = st.data_editor(
        st.session_state["plays_df"],
        column_config={
            "name": st.column_config.TextColumn("Play", required=True),
            "description": st.column_config.TextColumn(
                "Description", width="large"
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,
        key="plays_editor",
    )
    st.session_state["plays_df"] = edited

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button("💾 Save changes", type="primary", use_container_width=True):
            try:
                n = save_plays(st.session_state["plays_header"], edited)
                st.success(f"Saved {n} plays to {PLAYS_FILE.as_posix()}.")
            except Exception as exc:
                st.error(f"Failed to save: {exc}")
    with col_b:
        if st.button("↩️ Reload from file", use_container_width=True):
            hdr, df = load_plays()
            st.session_state["plays_header"] = hdr
            st.session_state["plays_df"] = df
            st.rerun()
    with col_c:
        st.caption(f"{len(edited)} plays currently in the catalog.")


def _render_predict_tab() -> None:
    st.subheader("Predict a play from images")
    st.warning("🚧 **Coming soon** — no trained model is available yet.")

    uploads = st.file_uploader(
        "Upload 1-5 frames from the possession",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="predict_uploads",
    )

    if uploads:
        if len(uploads) > 5:
            st.warning("Only the first 5 frames will be used.")
        cols = st.columns(min(len(uploads), 5))
        for i, f in enumerate(uploads[:5]):
            with cols[i]:
                st.image(f, use_container_width=True, caption=f"Frame {i + 1}")

    if st.button(
        "🔮 Predict play",
        type="primary",
        disabled=not uploads,
        use_container_width=True,
    ):
        plays_df: pd.DataFrame = st.session_state["plays_df"]
        result = predict_play_stub(
            uploads or [], plays_df["name"].dropna().tolist()
        )
        if result["status"] == "coming_soon":
            st.info(result["message"])
            st.caption(
                f"Received {result['image_count']} frame(s); vocabulary size "
                f"{result['vocab_size']}."
            )
        else:
            st.success("Prediction complete.")
            st.json(result)
