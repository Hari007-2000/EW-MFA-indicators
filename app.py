"""
EW-MFA Indicators & Decision Support — a standalone Streamlit app for the
Acetaminophen (PIOT Model D) manufacturing network.

Three pages:
  1. EW-MFA Indicators      — upload PIOT + Imports + Exports, compute the 8
                              direct indicators, each with its formula.
  2. Leontief Indicators    — A matrix / Leontief-inverse primer, then the 4
                              upstream (footprint) indicators with references.
  3. Decision Support       — a Low / Medium / High heatmap turning every
                              indicator into an industry-level decision.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import ewmfa_model as em

st.set_page_config(page_title="EW-MFA Indicators", page_icon="🧭",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      div[data-testid="stMetricValue"] {font-size: 1.4rem;}
      .ind-card {border:1px solid rgba(128,128,128,.25); border-radius:12px;
                 padding:16px 18px; margin-bottom:6px;}
      .muted {color:#6b7280; font-size:0.86rem;}
      .lvl-High{color:#c0392b; font-weight:700;}
      .lvl-Medium{color:#c47f0f; font-weight:700;}
      .lvl-Low{color:#2e7d32; font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

_HERE = os.path.dirname(__file__)
def _first_existing(*names):
    for n in names:
        p = os.path.join(_HERE, n)
        if os.path.exists(p):
            return p
    return os.path.join(_HERE, names[0])


SAMPLE = {
    "piot": _first_existing("PIOT_ModelD_APAP_workshop_final.csv",
                            "PIOT_ModelD_APAP_workshop.csv"),
    "imports": os.path.join(_HERE, "imports_apap.csv"),
    "exports": os.path.join(_HERE, "exports_apap.csv"),
    "de": os.path.join(_HERE, "domestic_extraction.csv"),
    "ri": os.path.join(_HERE, "resource_intensity.csv"),
}
LEVEL_COLORS = {"Low": "#2e7d32", "Medium": "#f39c12", "High": "#c0392b"}


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    a = abs(v)
    for d, s in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")]:
        if a >= d:
            return f"{v/d:,.2f}{s}"
    return f"{v:,.3g}"


# --------------------------------------------------------------------------- #
# Shared input / compute helpers (state persists across pages)
# --------------------------------------------------------------------------- #
def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def run_compute(piot_b, imp_b, exp_b, source_label, de_b=None, ri_b=None):
    out = em.compute_indicators(piot_b, imp_b, exp_b, de_b, ri_b)
    st.session_state["computed"] = out
    st.session_state["source_label"] = source_label


def has_results() -> bool:
    return "computed" in st.session_state


def results() -> dict:
    return st.session_state["computed"]


def indicator_bar(series: pd.Series, meta: dict) -> alt.Chart:
    """Horizontal bar chart for one indicator (PTB diverges red/blue)."""
    dfc = (series.rename("value").reset_index().rename(columns={"index": "Industry"}))
    if dfc.columns[0] != "Industry":
        dfc = dfc.rename(columns={dfc.columns[0]: "Industry"})
    unit = meta["unit"]
    if meta.get("diverging"):
        dfc["sign"] = np.where(dfc["value"] >= 0, "Net importer", "Net exporter")
        color = alt.Color("sign:N", title=None,
                          scale=alt.Scale(domain=["Net importer", "Net exporter"],
                                          range=["#c0392b", "#3E6E8E"]))
    else:
        color = alt.value("#3E6E8E")
    return (
        alt.Chart(dfc)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=f"{meta['name']} ({unit})", axis=alt.Axis(format="~s")),
            y=alt.Y("Industry:N", sort="-x", title=None),
            color=color,
            tooltip=["Industry", alt.Tooltip("value:Q", title=meta["key"], format=",.4g")],
        )
        .properties(height=26 * len(dfc) + 30)
    )


def _human(num):
    if num == 0:
        return "0"
    sign = "-" if num < 0 else ""
    n = abs(num)
    for div, suf, dec in [(1e9, "B", 2), (1e6, "M", 2), (1e3, "k", 1)]:
        if n >= div:
            return f"{sign}{n/div:.{dec}f} {suf}"
    return f"{sign}{n:.0f}"


def ptb_symlog_figure(series: pd.Series):
    """Diverging symmetric-log PTB chart, reproducing the notebook axis."""
    s = series.sort_values(ascending=False)
    labels = list(s.index)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ["#c0392b" if v >= 0 else "#3E6E8E" for v in s]   # red importer / blue exporter
    ax.barh(labels, s.values, color=colors)
    ax.axvline(0, linewidth=0.8, color="black")
    ax.set_xscale("symlog", linthresh=1)
    m = float(np.abs(s.values).max())
    if m > 0:
        lim = 10 ** math.ceil(math.log10(m))
        ax.set_xlim(-lim * 4, lim * 4)      # symmetric headroom for value labels
    for yi, val in enumerate(s.values):     # value labels just past each bar tip
        if val >= 0:
            ax.text(val, yi, "  " + _human(val), va="center", ha="left", fontsize=8.5)
        elif val < 0:
            ax.text(val, yi, _human(val) + "  ", va="center", ha="right", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Imports − Exports (kg/yr)   |   ← net exporter        net importer →")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def render_indicator(meta: dict, series: pd.Series, n: int) -> None:
    st.markdown(f"#### {n}. {meta['name']} ({meta['key']})")
    st.latex(meta["formula"])
    st.markdown(f"{meta['description']}")
    st.caption(f"Reference: {meta['reference']}")
    if meta["key"] == "PTB":
        fig = ptb_symlog_figure(series)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.altair_chart(indicator_bar(series, meta), use_container_width=True)
    st.divider()


# =========================================================================== #
# PAGE 1 — 8 direct EW-MFA indicators
# =========================================================================== #
def page_direct():
    st.title("EW-MFA Indicators — APAP manufacturing network")
    st.markdown(
        "Economy-wide material-flow (EW-MFA) indicators computed directly "
        "from the Physical Input–Output Table (PIOT) and the traded-commodity "
        "files. Upload your three files (or use the bundled Acetaminophen sample) "
        "and press **Compute the Indicators**."
    )

    _mfa = os.path.join(_HERE, "mfa_overview.png")
    if os.path.exists(_mfa):
        with st.container(border=True):
            st.markdown("**What material-flow analysis (MFA) delivers in manufacturing**")
            lc, mc, rc = st.columns([1, 6, 1])
            mc.image(_mfa, use_container_width=True)
            st.caption("Reference: Eurostat, Economy-wide material flow accounts (EW-MFA).")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        up_piot = c1.file_uploader("PIOT CSV", type=["csv"], key="u_piot")
        up_imp = c2.file_uploader("Imports CSV", type=["csv"], key="u_imp")
        up_exp = c3.file_uploader("Exports CSV", type=["csv"], key="u_exp")
        use_sample = st.checkbox("Use the bundled Acetaminophen sample files", value=True,
                                 help="Uncheck to require your own uploads.")
        go = st.button("Compute the Indicators", type="primary")

    # Resource intensity (for the Leontief indicators) always comes from the
    # bundled resource-intensity file.
    ri_bundled = _read(SAMPLE["ri"]) if os.path.exists(SAMPLE["ri"]) else None

    if go:
        try:
            if use_sample and up_piot is None:
                run_compute(_read(SAMPLE["piot"]), _read(SAMPLE["imports"]),
                            _read(SAMPLE["exports"]), "bundled sample", ri_b=ri_bundled)
            else:
                if up_piot is None:
                    st.error("Please upload a PIOT CSV (or tick the sample box).")
                    st.stop()
                imp_b = up_imp.getvalue() if up_imp is not None else None
                exp_b = up_exp.getvalue() if up_exp is not None else None
                run_compute(up_piot.getvalue(), imp_b, exp_b, "your uploaded files",
                            ri_b=ri_bundled)
            st.success(f"Computed from {st.session_state['source_label']}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Computation failed: {exc}")

    if not has_results():
        st.info("No results yet — press **Compute the Indicators** above.", icon="🧮")
        return

    out = results()
    res = out["results"]
    st.caption(f"Source: {st.session_state.get('source_label','')} · "
               f"{len(out['industries'])} industries · spectral radius of A = "
               f"{out['spectral_radius']:.4f}")

    # PTB commodity resolution table (transparency)
    with st.expander("Physical Trade Balance — resolved main commodity per industry"):
        st.dataframe(out["ptb_table"].style.format(
            {"Imports (kg/yr)": "{:,.0f}", "Exports (kg/yr)": "{:,.0f}", "PTB (kg/yr)": "{:,.0f}"}),
            use_container_width=True)

    st.divider()
    for i, meta in enumerate(em.DIRECT_INDICATORS, start=1):
        render_indicator(meta, res[meta["key"]], i)

    st.download_button("Download all indicator values (CSV)",
                       res.to_csv().encode(), "ewmfa_indicators.csv", "text/csv")


# =========================================================================== #
# PAGE 2 — Leontief upstream indicators
# =========================================================================== #
def page_leontief():
    st.title("Leontief-based upstream indicators")
    st.markdown(
        "The eight direct indicators describe each industry at its own gate. "
        "The **Leontief inverse** extends the view to the *whole upstream network* "
        "— everything a unit of output pulls from every supplier, directly and "
        "indirectly."
    )
    with st.container(border=True):
        st.markdown("**From the transaction table to the upstream multiplier**")
        st.latex(r"A = Z\,\hat{x}^{-1} \qquad\Longrightarrow\qquad a_{ij}=\frac{Z_{ij}}{x_j}")
        st.markdown(
            "The technical-coefficient matrix $A$ normalises each transaction by the "
            "receiving industry's total output $x_j$, so $a_{ij}$ is the material from "
            "industry $i$ needed to make one unit of $j$."
        )
        st.latex(r"L = (I - A)^{-1} = I + A + A^{2} + A^{3} + \cdots")
        st.markdown(
            "The **Leontief inverse** $L$ sums the entire chain of requirements: the "
            "direct input ($A$), the inputs needed to make those inputs ($A^{2}$), and "
            "so on. Its entry $l_{ij}$ is the total output of $i$ (direct **plus** "
            "indirect) required per unit of final demand for $j$. Weighting $L$ by a "
            "per-unit resource or waste intensity turns it into an **upstream impact "
            "assessment** — the footprint embodied across the supply chain, not just "
            "at the final stage."
        )

    if not has_results():
        st.warning("Compute the indicators first on the **EW-MFA Indicators** page, "
                   "then return here.", icon="⬅️")
        if st.button("Calculate now from the bundled sample"):
            run_compute(_read(SAMPLE["piot"]), _read(SAMPLE["imports"]),
                        _read(SAMPLE["exports"]), "bundled sample",
                        ri_b=(_read(SAMPLE["ri"]) if os.path.exists(SAMPLE["ri"]) else None))
            st.rerun()
        return

    if not st.button("Calculate the Leontief indicators", type="primary"):
        st.info("Press **Calculate the Leontief indicators** to draw the four figures.",
                icon="📈")
        # still show them by default once computed:
    out = results()
    res = out["results"]
    st.divider()
    for i, meta in enumerate(em.LEONTIEF_INDICATORS, start=1):
        render_indicator(meta, res[meta["key"]], i)


# =========================================================================== #
# PAGE 3 — Decision support heatmap
# =========================================================================== #
def page_decision():
    st.title("Decision support — indicator heatmap")
    st.markdown(
        "Each indicator is ranked across the industries and translated into a "
        "**decision**: Low, Medium or High priority for the action that indicator "
        "implies. Ranking uses within-network percentiles (oriented so **High** "
        "always means *most action needed*), cut at the 34th and 67th percentiles."
    )

    if not has_results():
        st.warning("Compute the indicators first on the **EW-MFA Indicators** page.",
                   icon="⬅️")
        if st.button("Calculate now from the bundled sample"):
            run_compute(_read(SAMPLE["piot"]), _read(SAMPLE["imports"]),
                        _read(SAMPLE["exports"]), "bundled sample",
                        ri_b=(_read(SAMPLE["ri"]) if os.path.exists(SAMPLE["ri"]) else None))
            st.rerun()
        return

    out = results()
    res = out["results"]
    labels, scores = em.decision_matrix(res)

    # Build long frame: rows = "KEY — decision question", cols = industry
    order_keys = [m["key"] for m in em.ALL_INDICATORS]
    row_label = {m["key"]: f"{m['key']} — {m['decision']}" for m in em.ALL_INDICATORS}
    long = []
    for key in order_keys:
        for ind in res.index:
            lvl = labels.loc[ind, key]
            long.append({
                "Decision": row_label[key],
                "Industry": ind,
                "Level": str(lvl),
                "Value": res.loc[ind, key],
                "keyorder": order_keys.index(key),
            })
    long = pd.DataFrame(long)
    row_sort = [row_label[k] for k in order_keys]

    base = alt.Chart(long)
    heat = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("Industry:N", title=None, axis=alt.Axis(labelAngle=-45, labelFontSize=11)),
        y=alt.Y("Decision:N", sort=row_sort, title=None,
                axis=alt.Axis(labelLimit=360, labelFontSize=11)),
        color=alt.Color("Level:N",
                        scale=alt.Scale(domain=["Low", "Medium", "High"],
                                        range=[LEVEL_COLORS["Low"], LEVEL_COLORS["Medium"],
                                               LEVEL_COLORS["High"]]),
                        legend=alt.Legend(title="Priority", orient="top")),
        tooltip=["Industry", "Decision",
                 alt.Tooltip("Value:Q", title="indicator value", format=",.4g"), "Level"],
    )
    text = base.mark_text(baseline="middle", fontSize=9, color="white", fontWeight="bold").encode(
        x=alt.X("Industry:N"), y=alt.Y("Decision:N", sort=row_sort),
        text="Level:N",
    )
    st.altair_chart((heat + text).properties(
        height=44 * len(order_keys) + 20,
        title="Decision priority by indicator (rows) and industry (columns)"
    ), use_container_width=True)

    st.caption("Green = Low priority · Amber = Medium · Red = High priority for the "
               "action the indicator implies.")

    # Decision dictionary
    st.markdown("#### What each indicator decides")
    dd = pd.DataFrame([
        {"Indicator": f"{m['name']} ({m['key']})",
         "Decision question": m["decision"],
         "High priority when": ("value is HIGH" if m["direction"] > 0 else "value is LOW")}
        for m in em.ALL_INDICATORS
    ])
    st.dataframe(dd, use_container_width=True, hide_index=True)

    # Per-industry high-priority summary
    st.markdown("#### High-priority actions by industry")
    hi = {}
    for ind in res.index:
        flagged = [f"{k} ({em.META_BY_KEY[k]['decision']})"
                   for k in order_keys if str(labels.loc[ind, k]) == "High"]
        hi[ind] = "; ".join(flagged) if flagged else "—"
    hi_df = pd.DataFrame({"Industry": list(hi.keys()), "High-priority decisions": list(hi.values())})
    st.dataframe(hi_df, use_container_width=True, hide_index=True)

    st.download_button("Download decision matrix (CSV)",
                       labels.to_csv().encode(), "ewmfa_decision_matrix.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🧭 EW-MFA Indicators")
st.sidebar.caption("Material-flow indicators & decision support for the "
                   "Acetaminophen (PIOT Model D) network")
if has_results():
    st.sidebar.success(f"Indicators computed from {st.session_state.get('source_label','')}.")
else:
    st.sidebar.info("Start on **EW-MFA Indicators** and press Compute.")

pages = [
    st.Page(page_direct, title="EW-MFA Indicators", icon="📊", default=True),
    st.Page(page_leontief, title="Leontief Indicators", icon="🔗"),
    st.Page(page_decision, title="Decision Support", icon="🧭"),
]
st.navigation(pages).run()
