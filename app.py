"""
EW-MFA Indicators & Decision Support — multi-page Streamlit app.

Pages:
  1. EW-MFA Indicators  — Physical Trade Balance (Imports + Exports only) AND the
                          direct PIOT indicators (CSID, SMIR, WGI, PWPR, MUE, MIU),
                          together on one page.
  2. Leontief Indicators — the 4 upstream (footprint) indicators: RF, WM, BL, URS.
  3. Decision Support    — Low / Medium / High decision heatmap.

Physical Trade Balance is computed only from the trade files (no PIOT) and is
independent of the other indicators. Everything is generic for any network.

Requires ewmfa_model.py (same folder). Run with:  streamlit run app.py
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
    "<style>.block-container{padding-top:2.2rem;padding-bottom:3rem;}"
    "div[data-testid='stMetricValue']{font-size:1.4rem;}</style>",
    unsafe_allow_html=True)

# --- Safety net: make an out-of-date module obvious instead of a cryptic error ---
for _fn in ("compute_ptb", "compute_indicators", "decision_matrix"):
    if not hasattr(em, _fn):
        st.error(
            f"Your **ewmfa_model.py** is out of date — it does not define `{_fn}`. "
            "Replace ewmfa_model.py with the version shipped alongside this app.py "
            "(both files must come from the same bundle), then restart the app."
        )
        st.stop()

_HERE = os.path.dirname(os.path.abspath(__file__))


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


def _align_to_industries(src: pd.Series, industries) -> pd.Series:
    """
    Map a commodity-indexed series (e.g. PTB) onto the PIOT industry index by
    name: exact → case-insensitive → unique substring. Generic for any network;
    unmatched industries get NaN.
    """
    exact = {str(k): v for k, v in src.items()}
    lower = {str(k).strip().lower(): v for k, v in src.items()}
    out = {}
    for ind in industries:
        name = str(ind)
        if name in exact:
            out[ind] = exact[name]
            continue
        key = name.strip().lower()
        if key in lower:
            out[ind] = lower[key]
            continue
        hits = [v for k, v in lower.items() if key and (key in k or k in key)]
        out[ind] = hits[0] if len(hits) == 1 else float("nan")
    return pd.Series(out, dtype="float64")


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
    s = ptb_df["PTB (kg/yr)"]
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n).sort_values(ascending=False)
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
    col = "#3E6E8E" if meta["key"] in ("CSID", "SMIR", "WGI", "PWPR", "MUE", "MIU") else "#4a6fa8"
    n = int(dfc["value"].notna().sum())
    return (
        alt.Chart(dfc.dropna(subset=["value"]))
        .mark_bar(color=col, size=26)                      # thicker bars
        .encode(
            x=alt.X("value:Q", title=f"{meta['name']} ({meta['unit']})",
                    axis=alt.Axis(format="~s", labelFontSize=13, titleFontSize=15,
                                  titleFontWeight="bold")),
            y=alt.Y("Industry:N", sort="-x", title=None,
                    axis=alt.Axis(labelFontSize=15, labelFontWeight="bold",
                                  labelLimit=320, labelPadding=6)),
            tooltip=["Industry", alt.Tooltip("value:Q", title=meta["key"], format=",.4g")],
        )
        .properties(height=42 * n + 60)                    # taller / wider chart area
    )


def render_indicator(meta: dict, series: pd.Series, n: int) -> None:
    st.markdown(f"#### {n}. {meta['name']} ({meta['key']})")
    st.latex(meta["formula"])
    st.markdown(meta["description"])
    st.caption(f"Reference: {meta['reference']}")
    st.altair_chart(indicator_bar(series, meta), use_container_width=True)
    st.divider()


def _run_indicators(piot_b, label):
    ri_b = _read(SAMPLE["ri"]) if os.path.exists(SAMPLE["ri"]) else None
    st.session_state["computed"] = em.compute_indicators(piot_b, ri_b)
    st.session_state["ind_source"] = label


def _restrict_ptb_to_industries(ptb_full: pd.DataFrame, industries) -> pd.DataFrame:
    """Restrict a full (union) PTB table to the PIOT's commodities, matched by
    name. Done here in app.py so it works with any ewmfa_model.py version."""
    imp = _align_to_industries(ptb_full["Imports (kg/yr)"], industries).fillna(0.0)
    exp = _align_to_industries(ptb_full["Exports (kg/yr)"], industries).fillna(0.0)
    out = pd.DataFrame({
        "Imports (kg/yr)": imp, "Exports (kg/yr)": exp,
        "PTB (kg/yr)": (imp - exp).astype(float),
    })
    out.index.name = "Commodity"
    return out.sort_values("PTB (kg/yr)", ascending=False)


def _run_all(piot_b, imp_b, exp_b, label):
    """Compute the PIOT indicators AND the PTB together, over the same
    commodity set: PTB is restricted to the PIOT's industries, matched to the
    Imports/Exports files by name."""
    _run_indicators(piot_b, label)
    industries = st.session_state["computed"]["industries"]
    # Call compute_ptb WITHOUT the industries kwarg (works with any model
    # version), then restrict to the PIOT commodities here.
    ptb_full = em.compute_ptb(imp_b, exp_b)
    st.session_state["ptb"] = _restrict_ptb_to_industries(ptb_full, industries)
    st.session_state["ptb_source"] = label


# =========================================================================== #
# PAGE 1 — Physical Trade Balance + direct EW-MFA indicators
# =========================================================================== #
def page_home():
    st.title("EW-MFA Indicators — APAP manufacturing network")
    st.markdown(
        "**Economy-wide material flow accounting (EW-MFA)** is a standardised framework "
        "that traces the physical materials — raw inputs, products, wastes/residuals, and "
        "imports/exports — flowing through an economy or production network, all measured in "
        "mass units (kg/yr). From these flows it derives a small set of headline indicators "
        "on resource efficiency, waste, circularity and trade dependence that support "
        "resource-productivity and sustainability decisions "
        "(Eurostat, *Economy-wide material flow accounts — Handbook*, 2018). "
        "This app computes those indicators for a manufacturing network. You upload the "
        "**Imports**, **Exports** and **PIOT** files together; the **Physical Trade "
        "Balance** is then calculated for the commodities in the PIOT (matched to the "
        "trade files by name) and reported alongside the **other indicators** from the "
        "PIOT. Everything is generic for any network."
    )

    _mfa = os.path.join(_HERE, "mfa_overview.png")
    if os.path.exists(_mfa):
        with st.container(border=True):
            st.markdown("**What material-flow analysis (MFA) delivers in manufacturing**")
            lc, mc, rc = st.columns([1, 6, 1])
            mc.image(_mfa, use_container_width=True)

    # ---------------- Combined upload & compute --------------------------- #
    st.header("1 · Upload the data & compute")
    st.markdown("Upload the **Imports**, **Exports** and **PIOT** CSVs (or tick the "
                "sample box), then press **Compute**. The PIOT sets the commodity list; "
                "the Physical Trade Balance and every other indicator are computed over "
                "that same set of commodities.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        up_imp = c1.file_uploader("Imports CSV", type=["csv"], key="imp")
        up_exp = c2.file_uploader("Exports CSV", type=["csv"], key="exp")
        up_piot = c3.file_uploader("PIOT CSV", type=["csv"], key="piot")
        use_sample = st.checkbox("Use the bundled sample files (Imports, Exports & PIOT)",
                                 value=True, key="all_sample")
        if st.button("Compute indicators & PTB", type="primary"):
            try:
                if use_sample and up_imp is None and up_exp is None and up_piot is None:
                    pb = _read(SAMPLE["piot"])
                    ib, eb, lbl = _read(SAMPLE["imports"]), _read(SAMPLE["exports"]), "bundled sample"
                elif up_piot is not None and up_imp is not None and up_exp is not None:
                    pb, ib, eb, lbl = (up_piot.getvalue(), up_imp.getvalue(),
                                       up_exp.getvalue(), "your uploaded files")
                else:
                    st.error("Upload ALL THREE files — Imports, Exports and PIOT "
                             "(or tick the sample box).")
                    st.stop()
                _run_all(pb, ib, eb, lbl)
                st.success(f"Indicators and Physical Trade Balance computed from {lbl}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Computation failed: {exc}")

    if "computed" not in st.session_state:
        st.info("No results yet — provide the three files (or tick the sample box) and "
                "press **Compute indicators & PTB**.", icon="🧮")
        return

    out = st.session_state["computed"]
    res = out["results"]
    st.caption(f"Source: {st.session_state.get('ind_source','')} · "
               f"{len(out['industries'])} industries · spectral radius of A = "
               f"{out['spectral_radius']:.4f}")

    st.divider()

    # ---------------- Physical Trade Balance (over PIOT commodities) ------- #
    st.header("2 · Physical Trade Balance")
    st.markdown(em.PTB_META["description"])
    st.caption(f"Reference: {em.PTB_META['reference']}")
    st.latex(r"\mathrm{PTB}_c = \mathrm{IMP}_c - \mathrm{EXP}_c")
    st.markdown("Computed for the **PIOT's commodities**, matched to the Imports and "
                "Exports files by name.")
    if "ptb" in st.session_state:
        ptb = st.session_state["ptb"]
        n_matched = int((ptb[["Imports (kg/yr)", "Exports (kg/yr)"]].abs().sum(axis=1) > 0).sum())
        st.caption(f"{len(ptb)} PIOT commodities · {n_matched} matched to the trade files.")
        n_show = st.slider("Show top N commodities by |PTB|", 5, max(6, len(ptb)),
                           min(25, len(ptb)), key="ptb_topn")
        st.pyplot(ptb_symlog_figure(ptb, n_show))
        st.caption("Red = net importer (PTB > 0) · Blue = net exporter (PTB < 0). Symmetric-log axis.")
        st.dataframe(ptb.style.format({"Imports (kg/yr)": "{:,.0f}", "Exports (kg/yr)": "{:,.0f}",
                                       "PTB (kg/yr)": "{:,.0f}"}), use_container_width=True)
        st.download_button("Download PTB (CSV)", ptb.to_csv().encode(),
                           "physical_trade_balance.csv", "text/csv")

    st.divider()

    # ---------------- Direct EW-MFA indicators (from the PIOT) ------------- #
    st.header("3 · EW-MFA Indicators (from the PIOT)")
    for i, meta in enumerate(em.DIRECT_INDICATORS, start=1):
        render_indicator(meta, res[meta["key"]], i)
    st.download_button("Download all indicator values (CSV)",
                       res.to_csv().encode(), "ewmfa_indicators.csv", "text/csv")
    st.info("Now open **Leontief Indicators** and **Decision Support** in the sidebar.",
            icon="➡️")


# =========================================================================== #
# PAGE 2 — Leontief upstream indicators
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
        st.markdown("The technical-coefficient matrix $A$ normalises each transaction by "
                    "the receiving industry's total output $x_j$.")
        st.latex(r"L = (I - A)^{-1} = I + A + A^{2} + A^{3} + \cdots")
        st.markdown(
            "Its entry $l_{ij}$ is the total output of $i$ (direct **plus** indirect) "
            "required per unit of final demand for $j$. Weighting $L$ by a per-unit "
            "resource intensity $r$ or waste intensity $w$ turns it into an **upstream "
            "impact assessment**. $r$ is taken from the resource-intensity file where a "
            "commodity matches, otherwise from a PIOT-derived primary intensity — so it "
            "is generic for any network."
        )

    if "computed" not in st.session_state:
        st.warning("Compute the indicators first on the **EW-MFA Indicators** page.", icon="⬅️")
        if st.button("Compute now from the bundled sample"):
            _run_all(_read(SAMPLE["piot"]), _read(SAMPLE["imports"]), _read(SAMPLE["exports"]), "bundled sample")
            st.rerun()
        return

    out = st.session_state["computed"]
    res = out["results"]
    st.caption(f"Resource-intensity source: {out.get('r_source','')}")
    st.divider()
    for i, meta in enumerate(em.LEONTIEF_INDICATORS, start=1):
        render_indicator(meta, res[meta["key"]], i)


# =========================================================================== #
# PAGE 3 — Decision support heatmap
# =========================================================================== #
def page_decision():
    st.title("Decision support — indicator heatmap")
    st.markdown(
        "Each PIOT indicator is ranked across the industries and translated into a "
        "decision: Low, Medium or High priority for the action it implies. Ranking uses "
        "within-network percentiles (oriented so **High** = most action needed), cut at "
        "the 34th and 67th percentiles. When the Physical Trade Balance has been computed "
        "it is included as the top row, ranked the same way (a large net import = High "
        "priority to secure domestic supply)."
    )
    if "computed" not in st.session_state:
        st.warning("Compute the indicators first on the **EW-MFA Indicators** page.", icon="⬅️")
        if st.button("Compute now from the bundled sample"):
            _run_all(_read(SAMPLE["piot"]), _read(SAMPLE["imports"]), _read(SAMPLE["exports"]), "bundled sample")
            st.rerun()
        return

    out = st.session_state["computed"]
    res = out["results"].copy()

    # --- Fold Physical Trade Balance in as an extra indicator column -------- #
    metas = list(em.ALL_INDICATORS)
    has_ptb = False
    if "ptb" in st.session_state:
        ptb_series = st.session_state["ptb"]["PTB (kg/yr)"]
        aligned = _align_to_industries(ptb_series, res.index)
        if aligned.notna().any():
            res.insert(0, "PTB", aligned.reindex(res.index))
            metas = [em.PTB_META] + metas
            has_ptb = True

    labels, _ = em.decision_matrix(res)

    order = [m["key"] for m in metas]
    row_label = {m["key"]: f"{m['key']} — {m['decision']}" for m in metas}
    rows = []
    for key in order:
        for ind in res.index:
            lvl = labels.loc[ind, key]
            if pd.isna(lvl):
                continue
            rows.append({"Decision": row_label[key], "Industry": ind,
                         "Level": str(lvl), "Value": res.loc[ind, key]})
    long = pd.DataFrame(rows)
    row_sort = [row_label[k] for k in order]

    base = alt.Chart(long)
    heat = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("Industry:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Decision:N", sort=row_sort, title=None, axis=alt.Axis(labelLimit=360)),
        color=alt.Color("Level:N", scale=alt.Scale(
            domain=["Low", "Medium", "High"],
            range=[LEVEL_COLORS["Low"], LEVEL_COLORS["Medium"], LEVEL_COLORS["High"]]),
            legend=alt.Legend(title="Priority", orient="top")),
        tooltip=["Industry", "Decision", alt.Tooltip("Value:Q", format=",.4g"), "Level"])
    text = base.mark_text(baseline="middle", fontSize=9, color="white", fontWeight="bold").encode(
        x="Industry:N", y=alt.Y("Decision:N", sort=row_sort), text="Level:N")
    st.altair_chart((heat + text).properties(height=44 * len(order) + 20),
                    use_container_width=True)
    st.caption("Green = Low · Amber = Medium · Red = High priority for the action the indicator implies.")

    if not has_ptb:
        st.info("Compute the Physical Trade Balance on the **EW-MFA Indicators** page to "
                "add it as the top row of this matrix.", icon="ℹ️")

    st.markdown("#### What each indicator decides")
    dd = pd.DataFrame([
        {"Indicator": f"{m['name']} ({m['key']})", "Decision question": m["decision"],
         "High priority when": ("value is HIGH" if m["direction"] > 0 else "value is LOW")}
        for m in metas])
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
else:
    st.sidebar.info("Start on **EW-MFA Indicators**: compute PTB, then the PIOT indicators.")

pages = [
    st.Page(page_home, title="EW-MFA Indicators", icon="📊", default=True),
    st.Page(page_leontief, title="Leontief Indicators", icon="🔗"),
    st.Page(page_decision, title="Decision Support", icon="🧭"),
]
st.navigation(pages).run()
