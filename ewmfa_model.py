"""
EW-MFA indicator engine for a Physical Input-Output Table (PIOT).

Replicates `PIOT_Model_D_Indicators_Colab_Final.ipynb`:

  * 8 direct EW-MFA indicators : PTB, MID, CSID, DIIS, WGI, PWPR, MUE, MIU
  * Leontief (upstream) indicators from L = (I - A)^-1, with A = Z x_hat^-1
      -> Resource Footprint (RF), Waste Multiplier (WM), Backward Linkage (BL),
         Upstream Resource Share (URS).   [Primary Resource Multiplier is
         computed internally but NOT reported, per request.]
  * A percentile -> Low / Medium / High decision engine per indicator.

No Streamlit / plotting here, so the engine is reusable and testable.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

RESERVED_COLS = {"ROE", "EXPORTS", "FINAL_DEMAND", "WASTE"}
RESERVED_ROWS = {"ROE", "IMPORTS", "SLACK"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_piot(source):
    """Read a PIOT CSV and auto-detect industries (any non-reserved column)."""
    df = pd.read_csv(source, index_col=0)
    df.index = [str(i).strip() for i in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    industries = [c for c in df.columns if c.upper() not in RESERVED_COLS]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df, industries


def read_trade_csv(source, value_keywords):
    """Read an Imports/Exports CSV keyed by commodity name (duplicates summed)."""
    frame = pd.read_csv(source)
    frame.columns = [str(c).strip() for c in frame.columns]
    commodity_col = frame.columns[0]
    value_col = None
    for c in frame.columns[1:]:
        if any(k in c.lower() for k in value_keywords):
            value_col = c
            break
    if value_col is None:
        value_col = frame.columns[1]
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    series = pd.Series(values.values, index=frame[commodity_col].astype(str).str.strip())
    return series.groupby(level=0).sum()


def _resolve_commodity(industry, trade_index):
    """Find the trade-file commodity representing an industry's main product."""
    low = {c.lower(): c for c in trade_index}
    if industry in trade_index:
        return industry
    if industry.lower() in low:
        return low[industry.lower()]
    hits = [c for c in trade_index if industry.lower() in c.lower()]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- #
# Accounting vectors
# --------------------------------------------------------------------------- #
def build_accounting(df: pd.DataFrame, industries: list[str]) -> dict:
    Z = df.loc[industries, industries].astype(float)

    II_in = Z.sum(axis=0)                                   # column sums
    self_use = pd.Series(np.diag(Z.to_numpy()), index=industries)
    cross_sector_in = II_in - self_use

    ROE_in = df.loc["ROE", industries].astype(float) if "ROE" in df.index \
        else pd.Series(0.0, index=industries)
    imports = df.loc["IMPORTS", industries].astype(float) if "IMPORTS" in df.index \
        else pd.Series(0.0, index=industries)
    TMI = II_in + ROE_in + imports

    II_out = Z.sum(axis=1)                                  # row sums
    ROE_out = df.loc[industries, "ROE"].astype(float) if "ROE" in df.columns \
        else pd.Series(0.0, index=industries)
    exports_col = df.loc[industries, "EXPORTS"].astype(float) if "EXPORTS" in df.columns \
        else pd.Series(0.0, index=industries)
    final_demand = df.loc[industries, "FINAL_DEMAND"].astype(float) if "FINAL_DEMAND" in df.columns \
        else pd.Series(0.0, index=industries)
    waste = df.loc[industries, "WASTE"].astype(float) if "WASTE" in df.columns \
        else pd.Series(0.0, index=industries)

    PROD = II_out + ROE_out + exports_col + final_demand
    TO = PROD + waste

    return dict(Z=Z, II_in=II_in, self_use=self_use, cross_sector_in=cross_sector_in,
                ROE_in=ROE_in, imports=imports, TMI=TMI, II_out=II_out, ROE_out=ROE_out,
                exports_col=exports_col, final_demand=final_demand, waste=waste,
                PROD=PROD, TO=TO, industries=industries)


def _safe_div(n, d, industries):
    n = pd.Series(n, index=industries, dtype=float)
    d = pd.Series(d, index=industries, dtype=float)
    return n.div(d.replace(0, np.nan))


# --------------------------------------------------------------------------- #
# Full computation
# --------------------------------------------------------------------------- #
def compute_indicators(piot_bytes: bytes, imports_bytes: bytes | None,
                       exports_bytes: bytes | None) -> dict:
    """
    Compute the 8 direct + 4 Leontief indicators.

    Returns a dict with:
      results   : DataFrame [industry x indicator]  (12 columns)
      ptb_table : DataFrame with resolved main commodity + imports/exports/PTB
      accounting: DataFrame of the PIOT accounting vectors
      industries: list
    """
    df, industries = load_piot(io.BytesIO(piot_bytes))
    acc = build_accounting(df, industries)
    TMI, PROD, waste = acc["TMI"], acc["PROD"], acc["waste"]
    imports, cross_sector_in, II_in = acc["imports"], acc["cross_sector_in"], acc["II_in"]

    # ----- Physical Trade Balance from the trade files (main product) ------- #
    if imports_bytes is not None and exports_bytes is not None:
        impL = read_trade_csv(io.BytesIO(imports_bytes), ["import"])
        expL = read_trade_csv(io.BytesIO(exports_bytes), ["export"])
        trade_index = sorted(set(impL.index) | set(expL.index))
        resolved, imp_vals, exp_vals = {}, {}, {}
        for ind in industries:
            c = _resolve_commodity(ind, trade_index)
            resolved[ind] = c if c is not None else "(not found)"
            imp_vals[ind] = float(impL.get(c, 0.0)) if c else 0.0
            exp_vals[ind] = float(expL.get(c, 0.0)) if c else 0.0
        imports_ptb = pd.Series(imp_vals, index=industries)
        exports_ptb = pd.Series(exp_vals, index=industries)
    else:
        # fall back to the PIOT's own EXPORTS column and IMPORTS row
        resolved = {i: i for i in industries}
        imports_ptb = imports.copy()
        exports_ptb = acc["exports_col"].copy()

    PTB = (imports_ptb - exports_ptb).astype(float)

    # ----- The other seven direct indicators -------------------------------- #
    MID = _safe_div(imports, TMI, industries) * 100
    CSID = _safe_div(cross_sector_in, TMI, industries) * 100
    DIIS = _safe_div(II_in, TMI, industries) * 100
    WGI = _safe_div(waste, TMI, industries) * 100
    PWPR = _safe_div(waste, PROD, industries)
    MUE = _safe_div(PROD, TMI, industries) * 100
    MIU = _safe_div(TMI, PROD, industries)

    # ----- Leontief (upstream) indicators ----------------------------------- #
    x = acc["TO"].to_numpy(dtype=float)
    inv_x = np.divide(1.0, x, out=np.zeros_like(x), where=x != 0)
    Zn = acc["Z"].to_numpy(dtype=float)
    A = Zn * inv_x[np.newaxis, :]
    L = np.linalg.inv(np.eye(len(industries)) - A)

    r = (imports.to_numpy() + acc["ROE_in"].to_numpy()) * inv_x      # primary-resource intensity
    w = waste.to_numpy() * inv_x                                     # waste intensity
    y = (acc["final_demand"] + acc["exports_col"] + acc["ROE_out"]).to_numpy(dtype=float)

    PRM = pd.Series(r @ L, index=industries)                        # computed, not reported
    BL = pd.Series(L.sum(axis=0), index=industries)
    WM = pd.Series(w @ L, index=industries)
    RF = PRM * pd.Series(y, index=industries)
    URS = _safe_div(PRM - pd.Series(r, index=industries), PRM, industries) * 100

    results = pd.DataFrame({
        "PTB": PTB, "MID": MID, "CSID": CSID, "DIIS": DIIS,
        "WGI": WGI, "PWPR": PWPR, "MUE": MUE, "MIU": MIU,
        "RF": RF, "WM": WM, "BL": BL, "URS": URS,
    })
    results.index.name = "Industry"

    ptb_table = pd.DataFrame({
        "Main commodity": [resolved.get(i, i) for i in industries],
        "Imports (kg/yr)": imports_ptb, "Exports (kg/yr)": exports_ptb,
        "PTB (kg/yr)": PTB,
    }, index=industries)

    accounting = pd.DataFrame({
        "II_in": acc["II_in"], "Self_use": acc["self_use"],
        "Cross_sector_in": cross_sector_in, "ROE_in": acc["ROE_in"],
        "Imports": imports, "TMI": TMI, "II_out": acc["II_out"],
        "ROE_out": acc["ROE_out"], "Exports": acc["exports_col"],
        "Final_demand": acc["final_demand"], "PROD": PROD,
        "Waste": waste, "Total_output": acc["TO"],
    })

    return dict(results=results, ptb_table=ptb_table, accounting=accounting,
                industries=industries, spectral_radius=float(np.max(np.abs(np.linalg.eigvals(A)))))


# --------------------------------------------------------------------------- #
# Indicator metadata: formula, unit, description, reference, decision question
# --------------------------------------------------------------------------- #
# direction: +1  -> a HIGH value means HIGH priority/need for the decision
#            -1  -> a LOW value means HIGH priority/need for the decision
DIRECT_INDICATORS = [
    dict(key="PTB", name="Physical Trade Balance", unit="kg/yr", diverging=True, direction=+1,
         formula=r"\mathrm{PTB}_j = \mathrm{IMP}_j - \mathrm{EXP}_j",
         description="Net physical trade on the industry's main traded product. "
                     "Positive = net importer; negative = net exporter.",
         reference="Eurostat (2018), Economy-wide material flow accounts handbook.",
         decision="Secure domestic supply / onshore?"),
    dict(key="MID", name="Material Import Dependency", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{MID}_j = \dfrac{\mathrm{IMP}_j}{\mathrm{TMI}_j}\times 100",
         description="Share of an industry's total material input that is met by imports.",
         reference="Eurostat (2018); OECD (2008), Measuring material flows.",
         decision="Reduce import dependency / diversify suppliers?"),
    dict(key="CSID", name="Cross-Sector Input Dependency", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{CSID}_j = \dfrac{\sum_{i\neq j} Z_{ij}}{\mathrm{TMI}_j}\times 100",
         description="Share of total inputs supplied by OTHER modelled industries "
                     "(diagonal self-use excluded).",
         reference="Miller & Blair (2009), Input-Output Analysis.",
         decision="Strengthen cross-sector supply coordination?"),
    dict(key="DIIS", name="Domestic Intermediate Input Share", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{DIIS}_j = \dfrac{\sum_{i} Z_{ij}}{\mathrm{TMI}_j}\times 100",
         description="Share of total inputs from domestic intermediate transactions "
                     "(includes diagonal self-use).",
         reference="Miller & Blair (2009).",
         decision="Manage internal supply concentration?"),
    dict(key="WGI", name="Waste Generation Intensity", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{WGI}_j = \dfrac{W_j}{\mathrm{TMI}_j}\times 100",
         description="Percentage of total material input that leaves the industry as waste.",
         reference="Eurostat (2018); Nakamura & Kondo (2009), Waste Input-Output.",
         decision="Need for closed-loop recovery?"),
    dict(key="PWPR", name="Physical Waste-to-Product Ratio", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{PWPR}_j = \dfrac{W_j}{\mathrm{PROD}_j}",
         description="Kilograms of waste generated per kilogram of non-waste product.",
         reference="Allwood et al. (2011), Material efficiency.",
         decision="Priority for waste minimisation / process redesign?"),
    dict(key="MUE", name="Material Utilization Efficiency", unit="%", diverging=False, direction=-1,
         formula=r"\mathrm{MUE}_j = \dfrac{\mathrm{PROD}_j}{\mathrm{TMI}_j}\times 100",
         description="Share of input mass converted into useful (non-waste) product.",
         reference="Allwood et al. (2011); OECD (2008).",
         decision="Need for process-efficiency improvement?"),
    dict(key="MIU", name="Material Intensity per Unit Product", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{MIU}_j = \dfrac{\mathrm{TMI}_j}{\mathrm{PROD}_j}",
         description="Kilograms of material input required per kilogram of product.",
         reference="Schmidt-Bleek (1993), MIPS concept.",
         decision="Reduce material intensity / feedstock burden?"),
]

LEONTIEF_INDICATORS = [
    dict(key="RF", name="Resource Footprint", unit="kg/yr", diverging=False, direction=+1,
         formula=r"\mathrm{RF}_j = \mathrm{PRM}_j \times y_j,\quad \mathrm{PRM}_j=\sum_i r_i\,l_{ij}",
         description="Primary material embodied in the final demand delivered by the "
                     "sector — direct plus all upstream requirements.",
         reference="Wiedmann et al. (2015), PNAS; Tukker et al. (2016), Glob. Env. Change.",
         decision="Manage upstream resource burden / footprint?"),
    dict(key="WM", name="Waste Multiplier", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{WM}_j = \sum_i w_i\,l_{ij}",
         description="Total waste generated across the whole upstream chain per unit of "
                     "the sector's output.",
         reference="Nakamura & Kondo (2002), J. Ind. Ecol.; Duchin (1990).",
         decision="Target upstream (embodied) waste reduction?"),
    dict(key="BL", name="Backward Linkage", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{BL}_j = \sum_i l_{ij}",
         description="Column sum of the Leontief inverse — total output pulled from the "
                     "whole network per unit of final demand (depth of upstream reliance).",
         reference="Rasmussen (1956); Hirschman (1958), Strategy of Econ. Development.",
         decision="Prioritise supply-chain resilience (deep upstream reliance)?"),
    dict(key="URS", name="Upstream Resource Share", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{URS}_j = \left(\dfrac{\mathrm{PRM}_j - r_j}{\mathrm{PRM}_j}\right)\times 100",
         description="Share of the embodied primary resource that is drawn from UPSTREAM "
                     "sectors rather than the sector's own stage.",
         reference="Suh (2004), Ecol. Econ.; Udo de Haes et al. (2002).",
         decision="Secure upstream feedstock (not just the final stage)?"),
]

ALL_INDICATORS = DIRECT_INDICATORS + LEONTIEF_INDICATORS
META_BY_KEY = {d["key"]: d for d in ALL_INDICATORS}


# --------------------------------------------------------------------------- #
# Decision engine: percentile -> Low / Medium / High per indicator
# --------------------------------------------------------------------------- #
def decision_matrix(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every indicator column, rank industries by percentile, orient by the
    indicator's decision direction (so 1 = strongest need for action), and cut
    into Low / Medium / High.

    Returns (priority_labels, oriented_scores), both [industry x indicator-key].
    """
    labels = pd.DataFrame(index=results.index)
    scores = pd.DataFrame(index=results.index)
    for key in results.columns:
        meta = META_BY_KEY.get(key)
        direction = meta["direction"] if meta else +1
        pct = results[key].rank(pct=True)
        oriented = pct if direction > 0 else (1 - pct)
        scores[key] = oriented
        labels[key] = pd.cut(oriented, bins=[-0.01, 0.34, 0.67, 1.01],
                             labels=["Low", "Medium", "High"])
    return labels, scores
