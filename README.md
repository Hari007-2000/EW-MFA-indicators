# EW-MFA Indicators & Decision Support

A **standalone** Streamlit app (separate from the Cascading Impact Explorer) that
computes the economy-wide material-flow (EW-MFA) indicators for a Physical
Input–Output Table (PIOT) and turns them into industry-level decisions. It
reproduces `PIOT_Model_D_Indicators_Colab_Final.ipynb`.

## Three pages

1. **EW-MFA Indicators** — upload the PIOT, Imports and Exports CSVs (or use the
   bundled Acetaminophen sample) and press **Compute the Indicators**. Shows the
   8 direct indicators, each with its formula, description and reference:
   PTB, MID, CSID, DIIS, WGI, PWPR, MUE, MIU.
2. **Leontief Indicators** — a primer on the technical-coefficient matrix
   `A = Z x̂⁻¹` and the Leontief inverse `L = (I−A)⁻¹` as an upstream
   impact-assessment tool, then the 4 upstream indicators with formulas and
   references: Resource Footprint (RF), Waste Multiplier (WM), Backward Linkage
   (BL), Upstream Resource Share (URS). *(The Primary Resource Multiplier is
   computed internally to build RF and URS but is not reported, as requested.)*
3. **Decision Support** — a Low / Medium / High **heatmap**: every indicator is
   ranked across the industries (within-network percentile, oriented so *High =
   most action needed*, cut at the 34th/67th percentiles) and phrased as a
   decision, e.g. *WGI — Need for closed-loop recovery?* Rows are the decisions,
   columns the industries. Includes a decision dictionary and per-industry
   high-priority summary.

## Files

| File | Purpose |
|------|---------|
| `app.py` | The 3-page Streamlit app |
| `ewmfa_model.py` | Compute engine + indicator metadata (formulas, references, decision rules) |
| `PIOT_ModelD_APAP_workshop_final.csv` | Bundled sample PIOT |
| `imports_apap.csv`, `exports_apap.csv` | Bundled sample trade files |
| `requirements.txt` | Dependencies |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input file formats

- **PIOT CSV** — industries as both rows and columns, plus the `ROE`, `IMPORTS`,
  `SLACK` rows and the `ROE`, `EXPORTS`, `FINAL_DEMAND`, `WASTE` columns.
  Industries are auto-detected (any column that is not a reserved block).
- **Imports / Exports CSV** — first column is the commodity name; the value
  column is the first header containing "import"/"export" (else the 2nd column).
  Each industry's *main traded product* is matched by name (exact →
  case-insensitive → unique substring) to compute PTB.

## Method (from the notebook)

Accounting: `TMI = ΣᵢZᵢⱼ + ROEᵢₙ + IMP`, `PROD = ΣⱼZᵢⱼ + ROEₒᵤₜ + EXP + FD`,
`TO = PROD + W`. Direct indicators are ratios of these. Leontief indicators use
`A = Z·x̂⁻¹` with `x = TO`, `L = (I−A)⁻¹`, resource intensity `r = (IMP+ROEᵢₙ)/x`,
waste intensity `w = W/x`, and final demand `y = FD + EXP + ROEₒᵤₜ`.

## Deploy to Streamlit Community Cloud

Put all five files (plus `requirements.txt`) in a GitHub repo, then at
share.streamlit.io → **Create app** → main file path `app.py`. Since this is a
distinct app, deploy it from its own folder/repo (or a subfolder), separate from
the Cascading Impact Explorer.
