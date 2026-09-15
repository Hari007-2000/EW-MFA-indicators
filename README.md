# EW-MFA Indicators & Decision Support (multi-page Streamlit app)

Two files run the app: **`app.py`** (UI) and **`ewmfa_model.py`** (calculations).
Keep them together and from the SAME bundle — if you update one, update the other.
`app.py` shows a clear message if `ewmfa_model.py` is out of date.

## Files
- `app.py`, `ewmfa_model.py`
- `PIOT_ModelD_APAP_workshop_final.csv`, `imports_apap.csv`, `exports_apap.csv`,
  `resource_intensity.csv`  (bundled sample)
- `mfa_overview.png` (optional diagram), `requirements.txt`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages
1. **EW-MFA Indicators** — Physical Trade Balance (from Imports + Exports only, no PIOT)
   AND the direct PIOT indicators (CSID, SMIR, WGI, PWPR, MUE, MIU), on one page.
2. **Leontief Indicators** — RF, WM, BL, URS (from the Leontief inverse).
3. **Decision Support** — Low / Medium / High decision heatmap.

Physical Trade Balance is independent of the PIOT indicators, and both are generic
for any network (industries/commodities are read from the uploaded files).
