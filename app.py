"""
EW-MFA Indicators & Decision Support — a generic Streamlit app.

Flow:
  1. Physical Trade Balance  — upload Imports + Exports ONLY; PTB is computed and
                               plotted here, independently of the PIOT.
  2. EW-MFA Indicators       — upload the PIOT to compute the 6 direct indicators.
  3. Leontief Indicators     — 4 upstream (footprint) indicators from the PIOT.
  4. Decision Support        — Low / Medium / High decision heatmap (10 indicators).

Both Physical Trade Balance and the PIOT indicators are generic: they work for
any network (industries / commodities are read from the uploaded files).

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
      .muted {color:#6b7280; font-size:0.86rem;}
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
    "ri": os.path.join(_HERE, "resource_intensity.csv"),
}
LEVEL_COLORS = {"Low": "#2e7d32", "Medium": "#f39c12", "High": "#c0392b"}


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    for d, s in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")]:
        if a >= d:
            return f"{v/d:,.2f}{s}"
    return f"{v:,.3g}"


def unit_fmt(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if unit == "%":
        return f"{v:,.3f}%"
    return fmt(v) + ("" if unit in ("kg/yr", "") else f" {unit}")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _human(num):
    if num == 0:
        return "0"
    sign = "-" if num < 0 else ""
    n = abs(num)
    for div, suf, dec in [(1e9, "B", 2), (1e6, "M", 2), (1e3, "k", 1)]:
        if n >= div:
            return f"{sign}{n/div:.{dec}f} {suf}"
    return f"{sign}{n:.0f}"


def ptb_symlog_figure(ptb_df: pd.DataFrame, top_n: int = 25):
    """Diverging symmetric-log PTB chart over the top-N commodities by |PTB|."""
    s = ptb_df["PTB (kg/yr)"]
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
    s = s.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(s) + 1.2)))
    colors = ["#c0392b" if v >= 0 else "#3E6E8E" for v in s]
    ax.barh(s.index.astype(str), s.values, color=colors)
    ax.axvline(0, linewidth=0.8, color="black")
    ax.set_xscale("symlog", linthresh=1)
    m = float(np.abs(s.values).max()) if len(s) else 0.0
    if m > 0:
        lim = 10 ** math.ceil(math.log10(m))
        ax.set_xlim(-lim * 4, lim * 4)
    ax.invert_yaxis()
    ax.set_xlabel("Imports − Exports (kg/yr)   |   ← net exporter        net importer →")
    ax.grid(axis="x", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def indicator_bar(series: pd.Series, meta: dict) -> alt.Chart:
    dfc = series.rename("value").reset_index()
    dfc = dfc.rename(columns={dfc.columns[0]: "Industry"})
    return (
        alt.Chart(dfc.dropna(subset=["value"]))
        .mark_bar(color="#3E6E8E" if meta["key"] in ("CSID", "SMIR", "WGI", "PWPR", "MUE", "MIU")
                  else "#4a6fa8")
        .encode(
            x=alt.X("value:Q", title=f"{meta['name']} ({meta['unit']})", axis=alt.Axis(format="~s")),
            y=alt.Y("Industry:N", sort="-x", title=None),
            tooltip=["Industry", alt.Tooltip("value:Q", title=meta["key"], format=",.4g")],
        )
        .properties(height=26 * dfc["value"].notna().sum() + 30)
    )


def render_indicator(meta: dict, series: pd.Series, n: int) -> None:
    st.markdown(f"#### {n}. {meta['name']} ({meta['key']})")
    st.latex(meta["formula"])
    st.markdown(meta["description"])
    st.caption(f"Reference: {meta['reference']}")
    st.altair_chart(indicator_bar(series, meta), use_container_width=True)
    st.divider()


# =========================================================================== #
# PAGE 1 — Physical Trade Balance (Imports + Exports only, no PIOT)
# =========================================================================== #
def page_ptb():
    st.title("Physical Trade Balance")
    st.markdown(
        "The Physical Trade Balance is computed **only from the Imports and Exports "
        "files** — it does not use the PIOT and is independent of every other "
        "indicator. Upload the two trade files (or use the bundled sample) and press "
        "**Compute Physical Trade Balance**. It works for any network."
    )
    st.latex(r"\mathrm{PTB}_c = \mathrm{IMP}_c - \mathrm{EXP}_c")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        up_imp = c1.file_uploader("Imports CSV", type=["csv"], key="ptb_imp",
                                  help="First column = commodity name; a value column "
                                       "containing 'import' (else the 2nd column).")
        up_exp = c2.file_uploader("Exports CSV", type=["csv"], key="ptb_exp",
                                  help="First column = commodity name; a value column "
                                       "containing 'export' (else the 2nd column).")
        use_sample = st.checkbox("Use the bundled sample trade files", value=True,
                                 key="ptb_sample")
        go = st.button("Compute Physical Trade Balance", type="primary")

    if go:
        try:
            if use_sample and up_imp is None and up_exp is None:
                imp_b, exp_b, lbl = _read(SAMPLE["imports"]), _read(SAMPLE["exports"]), "bundled sample"
            else:
                if up_imp is None or up_exp is None:
                    st.error("Please upload BOTH an Imports and an Exports CSV "
                             "(or tick the sample box).")
                    st.stop()
                imp_b, exp_b, lbl = up_imp.getvalue(), up_exp.getvalue(), "your uploaded files"
            st.session_state["ptb"] = em.compute_ptb(imp_b, exp_b)
            st.session_state["ptb_source"] = lbl
            st.success(f"Physical Trade Balance computed from {lbl}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not compute PTB: {exc}")

    if "ptb" not in st.session_state:
        st.info("No Physical Trade Balance yet — press the button above.", icon="⚖️")
        return

    ptb = st.session_state["ptb"]
    m = em.PTB_META
    st.caption(f"Source: {st.session_state.get('ptb_source','')} · {len(ptb)} commodities")
    st.markdown(m["description"])
    st.caption(f"Reference: {m['reference']}")

    n_show = st.slider("Show top N commodities by |PTB|", 5, min(60, len(ptb)),
                       min(25, len(ptb)))
    st.pyplot(ptb_symlog_figure(ptb, n_show))
    st.caption("Red = net importer (PTB > 0) · Blue = net exporter (PTB < 0). "
               "Symmetric-log axis.")

    st.dataframe(
        ptb.style.format({"Imports (kg/yr)": "{:,.0f}", "Exports (kg/yr)": "{:,.0f}",
                          "PTB (kg/yr)": "{:,.0f}"}),
        use_container_width=True)
    st.download_button("Download PTB (CSV)", ptb.to_csv().encode(),
                       "physical_trade_balance.csv", "text/csv")


# =========================================================================== #
# Shared: compute the PIOT-based indicators
# =========================================================================== #
def _run_indicators(piot_b, label):
    ri_b = _read(SAMPLE["ri"]) if os.path.exists(SAMPLE["ri"]) else None
    st.session_state["computed"] = em.compute_indicators(piot_b, ri_b)
    st.session_state["ind_source"] = label


def _piot_uploader_block():
    with st.container(border=True):
        up_piot = st.file_uploader("PIOT CSV", type=["csv"], key="ind_piot",
                                   help="Industries as both rows and columns, plus the "
                                        "ROE / IMPORTS / SLACK rows and ROE / EXPORTS / "
                                        "FINAL_DEMAND / WASTE columns.")
        use_sample = st.checkbox("Use the bundled sample PIOT", value=True, key="ind_sample")
        go = st.button("Compute the Indicators", type="primary")
    if go:
        try:
            if use_sample and up_piot is None:
                _run_indicators(_read(SAMPLE["piot"]), "bundled sample")
            else:
                if up_piot is None:
                    st.error("Please upload a PIOT CSV (or tick the sample box).")
                    st.stop()
                _run_indicators(up_piot.getvalue(), "your uploaded PIOT")
            st.success(f"Indicators computed from {st.session_state['ind_source']}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Computation failed: {exc}")


# =========================================================================== #
# PAGE 2 — EW-MFA direct indicators (from the PIOT)
# =========================================================================== #
def page_direct():
    st.title("EW-MFA Indicators — from the PIOT")
    st.markdown(
        "Direct economy-wide material-flow indicators computed from the Physical "
        "Input–Output Table. Industries are auto-detected from the file, so this is "
        "generic for any network. (Physical Trade Balance is on its own page and is "
        "not part of these.)"
    )

    _mfa = os.path.join(_HERE, "mfa_overview.png")
    if os.path.exists(_mfa):
        with st.container(border=True):
            st.markdown("**What material-flow analysis (MFA) delivers in manufacturing**")
            lc, mc, rc = st.columns([1, 6, 1])
            mc.image(_mfa, use_container_width=True)
            st.caption("Reference: Eurostat, Economy-wide material flow accounts (EW-MFA).")

    _piot_uploader_block()

    if "computed" not in st.session_state:
        st.info("No results yet — press **Compute the Indicators** above.", icon="🧮")
        return

    out = st.session_state["computed"]
    res = out["results"]
    st.caption(f"Source: {st.session_state.get('ind_source','')} · "
               f"{len(out['industries'])} industries · spectral radius of A = "
               f"{out['spectral_radius']:.4f}")
    st.divider()
    for i, meta in enumerate(em.DIRECT_INDICATORS, start=1):
        render_indicator(meta, res[meta["key"]], i)
    st.download_button("Download all indicator values (CSV)",
                       res.to_csv().encode(), "ewmfa_indicators.csv", "text/csv")


# =========================================================================== #
# PAGE 3 — Leontief upstream indicators
# =========================================================================== #
def page_leontief():
    st.title("Leontief-based upstream indicators")
    st.markdown(
        "The direct indicators describe each industry at its own gate. The "
        "**Leontief inverse** extends the view to the whole upstream network — "
        "everything a unit of output pulls from every supplier, directly and indirectly."
    )
    with st.container(border=True):
        st.latex(r"A = Z\,\hat{x}^{-1} \qquad\Longrightarrow\qquad a_{ij}=\frac{Z_{ij}}{x_j}")
        st.markdown(
            "The technical-coefficient matrix $A$ normalises each transaction by the "
            "receiving industry's total output $x_j$."
        )
        st.latex(r"L = (I - A)^{-1} = I + A + A^{2} + A^{3} + \cdots")
        st.markdown(
            "Its entry $l_{ij}$ is the total output of $i$ (direct **plus** indirect) "
            "required per unit of final demand for $j$. Weighting $L$ by a per-unit "
            "resource intensity $r$ or waste intensity $w$ turns it into an **upstream "
            "impact assessment**. The resource intensity $r$ is taken from the "
            "resource-intensity file where a commodity matches, otherwise from a "
            "PIOT-derived primary intensity — so it is generic for any network."
        )

    if "computed" not in st.session_state:
        st.warning("Compute the indicators first on the **EW-MFA Indicators** page.",
                   icon="⬅️")
        if st.button("Compute now from the bundled sample"):
            _run_indicators(_read(SAMPLE["piot"]), "bundled sample")
            st.rerun()
        return

    out = st.session_state["computed"]
    res = out["results"]
    st.caption(f"Resource intensity source: {out.get('r_source','')}")
    st.divider()
    for i, meta in enumerate(em.LEONTIEF_INDICATORS, start=1):
        render_indicator(meta, res[meta["key"]], i)


# =========================================================================== #
# PAGE 4 — Decision support heatmap (10 PIOT indicators, no PTB)
# =========================================================================== #
def page_decision():
    st.title("Decision support — indicator heatmap")
    st.markdown(
        "Each PIOT indicator is ranked across the industries and translated into a "
        "decision: Low, Medium or High priority for the action it implies. Ranking uses "
        "within-network percentiles (oriented so **High** = most action needed), cut at "
        "the 34th and 67th percentiles. Physical Trade Balance is a standalone indicator "
        "and is not included here."
    )
    if "computed" not in st.session_state:
        st.warning("Compute the indicators first on the **EW-MFA Indicators** page.",
                   icon="⬅️")
        if st.button("Compute now from the bundled sample"):
            _run_indicators(_read(SAMPLE["piot"]), "bundled sample")
            st.rerun()
        return

    out = st.session_state["computed"]
    res = out["results"]
    labels, _ = em.decision_matrix(res)

    order = [m["key"] for m in em.ALL_INDICATORS]
    row_label = {m["key"]: f"{m['key']} — {m['decision']}" for m in em.ALL_INDICATORS}
    long = []
    for key in order:
        for ind in res.index:
            long.append({"Decision": row_label[key], "Industry": ind,
                         "Level": str(labels.loc[ind, key]),
                         "Value": res.loc[ind, key]})
    long = pd.DataFrame(long)
    row_sort = [row_label[k] for k in order]

    base = alt.Chart(long)
    heat = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("Industry:N", title=None, axis=alt.Axis(labelAngle=-45, labelFontSize=11)),
        y=alt.Y("Decision:N", sort=row_sort, title=None, axis=alt.Axis(labelLimit=360)),
        color=alt.Color("Level:N", scale=alt.Scale(
            domain=["Low", "Medium", "High"],
            range=[LEVEL_COLORS["Low"], LEVEL_COLORS["Medium"], LEVEL_COLORS["High"]]),
            legend=alt.Legend(title="Priority", orient="top")),
        tooltip=["Industry", "Decision",
                 alt.Tooltip("Value:Q", title="value", format=",.4g"), "Level"])
    text = base.mark_text(baseline="middle", fontSize=9, color="white", fontWeight="bold").encode(
        x="Industry:N", y=alt.Y("Decision:N", sort=row_sort), text="Level:N")
    st.altair_chart((heat + text).properties(height=44 * len(order) + 20),
                    use_container_width=True)
    st.caption("Green = Low · Amber = Medium · Red = High priority for the action the "
               "indicator implies.")

    st.markdown("#### What each indicator decides")
    dd = pd.DataFrame([
        {"Indicator": f"{m['name']} ({m['key']})", "Decision question": m["decision"],
         "High priority when": ("value is HIGH" if m["direction"] > 0 else "value is LOW")}
        for m in em.ALL_INDICATORS])
    st.dataframe(dd, use_container_width=True, hide_index=True)
    st.download_button("Download decision matrix (CSV)",
                       labels.to_csv().encode(), "ewmfa_decision_matrix.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🧭 EW-MFA Indicators")
st.sidebar.caption("Physical Trade Balance + material-flow indicators & decision support")
if "ptb" in st.session_state:
    st.sidebar.success("Physical Trade Balance computed.")
if "computed" in st.session_state:
    st.sidebar.success(f"Indicators computed from {st.session_state.get('ind_source','')}.")
st.sidebar.caption("Start on **Physical Trade Balance** (Imports + Exports), then go to "
                   "**EW-MFA Indicators** and upload the PIOT.")

pages = [
    st.Page(page_ptb, title="Physical Trade Balance", icon="⚖️", default=True),
    st.Page(page_direct, title="EW-MFA Indicators", icon="📊"),
    st.Page(page_leontief, title="Leontief Indicators", icon="🔗"),
    st.Page(page_decision, title="Decision Support", icon="🧭"),
]
st.navigation(pages).run()
