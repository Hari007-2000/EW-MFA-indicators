# EW-MFA Indicators & Decision Support — single-file Streamlit app

Everything is in **`app.py`** (model + UI + sample data embedded), so there is no
separate module to get out of sync.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## One page, two independent parts
1. **Physical Trade Balance** — upload Imports + Exports (or use the bundled sample).
   PTB = Imports − Exports per commodity. Uses ONLY the trade files, not the PIOT.
2. **EW-MFA Indicators (from the PIOT)** — upload the PIOT to compute the 6 direct
   indicators (CSID, SMIR, WGI, PWPR, MUE, MIU) and the 4 Leontief upstream
   indicators (RF, WM, BL, URS), plus a Low/Medium/High decision heatmap.

Both are generic for any network (industries/commodities are read from the files).
`mfa_overview.png` is optional (shown if present). Deploy needs only `app.py` and
`requirements.txt` (add `mfa_overview.png` if you want the diagram).
