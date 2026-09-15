"""
EW-MFA Indicators & Decision Support — single-file Streamlit app.

Everything (model + UI + sample data) is in this one file, so there is no
separate module to get out of sync. Physical Trade Balance and the PIOT
indicators live on ONE page.

  * Physical Trade Balance : from the Imports + Exports files only (no PIOT).
  * PIOT indicators        : CSID, SMIR, WGI, PWPR, MUE, MIU + Leontief RF, WM, BL, URS.
  * Decision heatmap       : Low / Medium / High per indicator.

Both are generic for any network. Run with:  streamlit run app.py
"""
from __future__ import annotations

import io, math, os, base64
import numpy as np
import pandas as pd
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="EW-MFA Indicators", page_icon="\U0001F9ED",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1150px;}
div[data-testid="stMetricValue"]{font-size:1.4rem;}</style>""", unsafe_allow_html=True)

LEVEL_COLORS = {"Low":"#2e7d32","Medium":"#f39c12","High":"#c0392b"}



# --------------------------------------------------------------------------- #
# Embedded bundled sample (Acetaminophen network), base64-encoded. Uploads override.
# --------------------------------------------------------------------------- #
_B64 = {
  "piot": "LEFjZXRhbWlub3BoZW4sQWNldGljIEFjaWQsQWNldGljIEFuaHlkcmlkZSxBY2V0b25lLEFtbW9uaWEsQmVuemVuZSxIeWRyb2dlbiBTTVIsSXNvIHByb3Bhbm9sLE1ldGhhbmUsTWV0aGFub2wsTmFwaHRoYSxOaXRyaWMgQWNpZCxOaXRybyBiZW56ZW5lLFBhcmEgYW1pbm9waGVub2wsUHJvcGVuZSxTdWxwaHVyaWMgYWNpZCxST0UsRVhQT1JUUyxGSU5BTF9ERU1BTkQsV0FTVEUKQWNldGFtaW5vcGhlbiwxMzEyNi4zMjUsMC4wLDU1NzU3NjMuMjE1LDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwzNjMwNC4wMDA1ODg0MTcwNSwxNjE0MTA1LjgzMyw4MDc1OTkwLjYyOCwyNTI2ODcxMC4wCkFjZXRpYyBBY2lkLDgxMDY4MzkuMzE1LDAuMCwzNDQzNjAwMzUzLjExNCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMzExNzY5NjAwLjAwMTA2MzM1LDc2MzkxODgxNC41MTMsNDIxODE2MDQxLjA1NywzNDMzNDY0ODMyLjAKQWNldGljIEFuaHlkcmlkZSw1MTQ4MTE2LjE0OSwwLjAsMjY3NTUyMzc3MS4zMjIsMC4wLDc3MzkxLjU0OCw5OTg2MTExLjUwMSw1Nzg0NDA3LjczLDAuMCwwLjAsNjkxMzMuNTc5LDAuMCwyMTgwMy4zMjEsMC4wLDAuMCwwLjAsMC4wLDExNDk5Mzk5MDUuMDQyMzIyNCwzOTg2MzMwODUuMTU3LDQxNDg0NjA3NC42NTEsNTY1NTYyNTMyMC4wCkFjZXRvbmUsMC4wLDAuMCw3NjY0MTkzMjcuNywwLjAsMjM5NDI2Ni4xNjQsMC4wLDAuMCwwLjAsMC4wLDU1LjgzMywwLjAsOTc0Ni4zNTYsMC4wLDM3LjgzNSwxMTkzNjcyLjI5NCwwLjAsMTkxMzUxOTUuODE0OTYxNTQsMzEzNzE3OTMuMTYsNTU2MjAxNjAuODQzLDExMTg5MDY4OC4wCkFtbW9uaWEsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwxNTUwMTM1ODYxLjUxMywwLjAsMTE2MDg0OS4xMTgsMC4wLDAuMCwxMTU3MjUwODI0LjAwMDgxMjUsOTgxMDE5MzUyLjE2NywxNTYwNzEyNjA1Ny4yMDIsMTUxMDA5NDQ4MC4wCkJlbnplbmUsMC4wLDAuMCwwLjAsMC4wLDQ4MzI2NDY0LjUxMywwLjAsMC4wLDAuMCwwLjAsMTEyNi45NTMsMC4wLDE5NjcyMi44ODEsODU0NTYxLjIxNiw3NjMuNjc4LDI0MDkzMzc4Ljg0NSwwLjAsMTU1ODQxODI1MzkuMjI3MzIsNDgwOTkyLjE0NywxNDc3NTc4MTguNTQsMzQ3NzUyNzA0NjQuMApIeWRyb2dlbiBTTVIsMC4wLDAuMCwwLjAsMC4wLDgzMDkzNTUzOC45MSwwLjAsNzg2Nzc5MzAyODUuOTIxLDAuMCwwLjAsMzU3MDI5MDkwMi43MDcsMC4wLDMzODI0OTUuMTY5LDAuMCwxMzEzMC44MzksNDE0MjY2Njk0Ljk5MSwwLjAsMjE5MzIxNTY5MjEzLjg2MDQ3LDI4MjQ2NDYuMzQ2LDIzNTM4NzE5NTUuMjU3LDIwNDAxODcxNDg0MC4wCklzbyBwcm9wYW5vbCwwLjAsMC4wLDAuMCw0ODk0ODMxNTAuNDc0LDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDE5ODE0NzA1MDYzLjk5OTYzOCwxNTg3NDM4MjMuMzYzLDE0NjM0MTk2Mi4xNjMsOTMzMDk1MTg0LjAKTWV0aGFuZSwwLjAsMC4wLDAuMCwwLjAsNzIxMDUzLjAyNiw5MzA0MDA4MC43NTgsMTk1NjExODE2My42MywxMzAwNTc3MS45NDMsMC4wLDg3NjQ1MDEwLjcyOSwwLjAsNDgwNDI2Ljg5LDAuMCwwLjAsMTY3MzY3NTE5MTkuMTcxLDAuMCwyNDIxODQzNzgyNDMuMzAwOTYsMTYxODY5MDg5LjY4NywxMTY0MzUzNzkyLjg2NSw3MDI1NTIyMDAwNjQuMApNZXRoYW5vbCwwLjAsNDU0MjE5NjYzMy4wMSwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMDAwMjc0NjU4MjAzMTI1LDI5ODc0MjIxMTQuMzU2LDI2NzI5NTY2MjguNjM0LDY2ODYwMTQyODg3Ljk5OTk5Ck5hcGh0aGEsMC4wLDAuMCwwLjAsMC4wLDYzMzk2LjkxLDgxODAzMzIuNjM5LDQ3Mzg0MTguODg3LDc3ODg3MC42MzMsMC4wLDE0NDc4My45MjQsMC4wLDUzNzI5Ljg0NiwwLjAsMC4wLDEyNjY2OTIzMDguNDY2LDAuMCwzMDM3NTMxMjg0NDEuMjU4MDYsNTkzODQ3NDM4Ny4zNTgsMzgzODEzMDY3NC4wNzksMC4wCk5pdHJpYyBBY2lkLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsNDYzMTIyMzYyMjMuOTEsNzQ0Nzc2MDkwLjc5OSwwLjAsMC4wLDIxNjk2LjA5LDQ4MDU0MDQxNTM1OS45OTg0LDExNzA5MzA5NC4zMjUsMTc1NjM5NjQxNC44NzYsMjM0MzQ1MTMyODAuMDAwMDA0Ck5pdHJvIGJlbnplbmUsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDEwNTI1NjAwLjc4NiwwLjAsMC4wLDAuMCwwLjAsMjA3Mzg3NzIuNDA5LDExMTA4OTIuMDc4LDAuMCwwLjAsMjAwNzg5MzYyLjQxOTQzMzkyLDg5NDA2Njg1LjE2MSwxNTYzMjY4NDYzLjE0NiwyNDI4MjQ1NDcyLjAKUGFyYSBhbWlub3BoZW5vbCw1MTE3OS42NjYsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMTM2NjIwNTUuOTk4NzAwMjgzLDQwMzEuMTY1LDMyMjQ5My4xNjksMjM5MDcyMC4wClByb3BlbmUsMC4wLDAuMCwwLjAsMC4wLDMwMzg1MzQ3MDYuMjE1LDUwNDQxNTY3ODkxLjA0NCwyOTIxODAzODk4OS45NjcsMzQ2MTUyMDcwNi43OTksMC4wLDM0OTI2NzM0Ny40OTMsMC4wLDEyMDkxMDAwMC44OTgsODEyMTA1MzgwLjY5MSw0MTgzOC45LDEzMTk5ODEzNDQuMjA3LDAuMCwxMTUxODU2OTU3MjIyOS41LDg4NzAwMTUwNjM1LjY5NSw2NTk3MzQxNTI2MDguNTkxLDYzNzI5NTAyNzU0NC4wClN1bHBodXJpYyBhY2lkLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCw3NTY1NjE3MDUuNTI0LDAuMCwwLjAsMC4wLDAuMCwxNDkwNjY2NTUxLjI2Nyw0Njk2MjEuMDgyLDAuMCwwLjAsNTMxMTE4MzU0NTMuNTgxLDM3ODgyOTg3My4yMTksMTM0Njk1MDY2MDMuMzI3LDY0NjAxNTY5Mjk2LjAwMDAxClJPRSwyMzA3MDA1OS40ODcsMzI4Mzk3ODYxMi41NDUsMzIyMDk5NzI4NC44MDMsNDcwODMwMTY4LjkxNSwxNjg4NDcwNzkyMC4wLDAuMCwzOTkzMTQ3NDQ0MjQuMCwxNzA5MzMwMjcyOC41NzgsOTY0OTUwNTYzNjE2LjAsNzMwNTUxMDMzMzcuODkxLDMxNDgxMDM4NTM0NC4wLDUwNDcyOTk1NTI2MC40MTEsODM1NzI4MDAwLjAsMTMzOTk1MTIuMCwxMjk3MzI5NzQ4OTc0MC4zNjMsMTMzODA5NDE3NDA3LjkxLDAuMCwxMzMxMjM3ODExLjQwMyw2MDA1NDcyMzQ1LjY5MywwLjAKSU1QT1JUUyw0MTk0Njc5LjA1ODQxMTU4Myw1NTY1MDEyMzQuNDQ1MDYzMSwyMDM1Mzg2MTkuODQ1Njc5MDcsMjc3MjE2MjQuNjEwOTYxNTM4LDEwMjY2ODYuNzEzMTg5MjExOCwyODM5MDQxNi4wNTc2Nzg0MzUsMTY0NDUwMTMuODY0NDg2MDYsMjA2NjczNzk5LjczNjYzNDU1LDAuMCwxOTY1NjQuODkxMjY2NjQ3ODQsMC4wLDE4ODA2OTg4OC44MDMzMzE5LDQwOTIxNTg5MS42MTc0MzM3LDIzMzgzNC40Njg3MDAyODM4NSw0MDIxNjUuNjYzNjI0MjU5LDAuMCwwLjAsMzI4NjkyOTkzNi43ODk2MDg1LDg1NTk0NTk2NDMuNDMzOTMxLApTTEFDSywsLCwsLCwsLCwsLCwsLCwsMjU1NzM5ODAxMzc4MS45OTc2LCwsCg==",
  "imp":  "Q29tbW9kaXR5LElNUE9SVFMKQWNldGljIEFjaWQsMTE0MDI4MzAuNDMKQWNldGljIEFjaWQgQnlwcm9kdWN0IDEsMApDYXJib24gTW9ub3hpZGUsMApNZXRoYW5vbCwxMTY1NDc3NzM4CkFjZXRpYyBBY2lkIEJ5cHJvZHVjdCAyLDAKQWNldGljIEFjaWQgV2FzdGUsMApBY2V0aWMgQWNpZCBCeXByb2R1Y3QgMywwCldhdGVyLDAKQWNldGljIEFuaHlkcmlkZSwyMTk5OTgxMS40CkFjZXRvbmUsNjc5OTYzMDgKTWV0aGFuZSwzMDI1MzM2MzE3MApBY2V0aWMgQW5oeWRyaWRlIEJ5cHJvZHVjdCwwCkFjZXRpYyBBbmh5ZHJpZGUgV2FzdGUsMApBY2V0b25lIEJ5cHJvZHVjdCAxLDAKSXNvIHByb3Bhbm9sLDczNzMyMDYxLjU0CkFjZXRvbmUgQnlwcm9kdWN0IDIsMApBY2V0b25lIFdhc3RlLDAKQW1tb25pYSw3MDQ5NTQ4NDMKQXJnb24sNDE2MTQyMjYuOQpBbW1vbmlhIEJ5cHJvZHVjdCAxLDAKQW1tb25pYSBFcXVpbGlicml1bSBSZWFjdG9yIEZlZWQsMApBbW1vbmlhIEVxdWlsaWJyaXVtIFJlYWN0b3IgUHJvZHVjdCwwCkh5ZHJvZ2VuIFNNUiwxMTQ1NTAyOS4yCkxpcXVpZCBJbXB1cmUgQW1tb25pYSwwCk5pdHJvZ2VuLDc3ODYzMgpBbGthbmUgR2FzIE1peHR1cmUsMApBbW1vbmlhIEdhcyBNaXh0dXJlLDAKQWlyIE1peHR1cmUsMApBbW1vbmlhIE91dHB1dCBHYXMgTWl4dHVyZSwwCkFtbW9uaWEgQnlwcm9kdWN0IDIsMApBbW1vbmlhIEJ5cHJvZHVjdCAzLDAKQW1tb25pYSBXYXN0ZSwwCkFjZXRhbWlub3BoZW4sMTkzMDM4OTQuMTMKUGFyYSBhbWlub3BoZW5vbCwxMDA0MTQuNwpBUEFQIFdhc3RlIDEsMApBUEFQIFdhc3RlIDIsMApCZW56ZW5lLDEyMDE1MjY3MjMKQ29rZSwxNDc0MzMzMTcKQmVuemVuZSBOYXBodGhhLDI0NzA4MDg5MjcKTmFwaHRoYWxlbmUsMApCZW56ZW5lIEJ5cHJvZHVjdCwwCkJlbnplbmUgV2FzdGUsMApBaXIsMApDYXJib24gRGlveGlkZSw0NTc1MTk2NS4xNQpIeWRyb2dlbiBCeXByb2R1Y3QsMApIeWRyb2dlbiBXYXN0ZSwwCklQQSBHYXMgTWl4dHVyZSwwClN1bHBodXJpYyBhY2lkLDE4NjY1NDQzNzUKUHJvcGFuZSwxMTM2NzkyMTEKUHJvcGVuZSwxOTg1ODk3ODgKSVBBIEJ5cHJvZHVjdCwwCklQQSBXYXN0ZSwwCkV0aGFuZSw1MDQwMDI3NjcuNQpNZXRoYW5lIEZlZWRnYXMsNjA1MDY3MjYzNDAKTWV0aGFuZSBHYXMgTWl4dHVyZSwwCk1ldGhhbmUgTGVhbiBHYXMsMApNZXRoYW5lIExlYW4gVEVHLDAKTWV0aGFuZSBSaWNoIFRFRywwCk1ldGhhbmUgQnlwcm9kdWN0IDEsMApNZXRoYW5lIEJ5cHJvZHVjdCAyLDAKTWV0aGFuZSBXYXN0ZSwwCk1ldGhhbm9sIEJ5cHJvZHVjdCAxLDAKTWV0aGFub2wgTml0cm9nZW5fV2F0ZXIgLDAKTWV0aGFub2wgQnlwcm9kdWN0IDIsMApNZXRoYW5vbCBXYXN0ZSwwCk1ldGhhbm9sIE1lT0hfRXRPSF9XYXRlciwwCk1ldGhhbm9sIEdhcyBNaXh0dXJlLDAKTml0cmljIEFjaWQsMjUxMjQ4OTQuMjkKTml0cm8gYmVuemVuZSwxMzQ2Mi4zMDc2OQpOaXRyb2JlbnplbmUgV2FzdGUsMApOaXRyaWMgQWNpZCBHYXMgTWl4dHVyZSAxLDAKTml0cmljIEFjaWQgV2F0ZXIgTWl4dHVyZSwwCk5pdHJpYyBBY2lkIEdhcyBNaXh0dXJlIDIsMApOaXRyaWMgQWNpZCBHYXMgTWl4dHVyZSAzLDAKTml0cmljIEFjaWQgQnlwcm9kdWN0LDAKTml0cmljIEFjaWQgV2FzdGUsMApBbmlsaW5lLDM3MzQyOTM0LjY3ClBBUCBXYXN0ZSAxLDAKUEFQIFdhc3RlIDIsMApQQVAgV2FzdGUgMywwClBBUCBXYXN0ZSA0LDAKQ3J1ZGUgQUdPLDAKQnV0YW5lLDEwOTg0MTY3MDUKRGllc2VsLDE3MzA3NzQ4LjEKQ3J1ZGUgR2FzIE1peHR1cmUsMApDcnVkZSBITmFwaHRoYSwwCktlcm9zZW5lLDAKTWl4Y3J1ZGUsNTI0Njg5OTI3ODIKTmFwaHRoYSwyNDcwODA4OTI3CkNydWRlIEJ5cHJvZHVjdCwwClBlbnRhbmUsMjQ0MTU3MC41NDcKQ3J1ZGUgV2FzdGUsMApQcm9wZW5lIEMySDZfSDIsMApQcm9wZW5lIENINF9IMiwwClByb3BlbmUgQnlwcm9kdWN0LDAKUHJvcGVuZSBXYXN0ZSwwClN1bGZ1cmljIEFjaWQgQnlwcm9kdWN0LDAKU3VsZnVyLDEwMDYwMDAyMTUKU3VsZnVyaWMgQWNpZCBXYXN0ZSwwCg==",
  "exp":  "Q29tbW9kaXR5LEVYUE9SVFMsRklOQUxfREVNQU5ECkFjZXRpYyBBY2lkLDYxNTMwNDA2My44LDAKQWNldGljIEFjaWQgQnlwcm9kdWN0IDEsMCwwCkNhcmJvbiBNb25veGlkZSwwLDAKTWV0aGFub2wsMTQxNzI2Njg4OCwwCkFjZXRpYyBBY2lkIEJ5cHJvZHVjdCAyLDAsMApBY2V0aWMgQWNpZCBXYXN0ZSwwLDAKQWNldGljIEFjaWQgQnlwcm9kdWN0IDMsMCwwCldhdGVyLDAsMApBY2V0aWMgQW5oeWRyaWRlLDYwNDA1MzQ1LjYxLDAKQWNldG9uZSw3MzkzODg0OSwwCk1ldGhhbmUsMjk1NDk0NDQ5NjAsMApBY2V0aWMgQW5oeWRyaWRlIEJ5cHJvZHVjdCwwLDAKQWNldGljIEFuaHlkcmlkZSBXYXN0ZSwwLDAKQWNldG9uZSBCeXByb2R1Y3QgMSwwLDAKSXNvIHByb3Bhbm9sLDE5NzIwNDEzNS45LDAKQWNldG9uZSBCeXByb2R1Y3QgMiwwLDAKQWNldG9uZSBXYXN0ZSwwLDAKQW1tb25pYSw5NjgyNjMyMCwwCkFyZ29uLDE4NTAzODY3LjMxLDAKQW1tb25pYSBCeXByb2R1Y3QgMSwwLDAKQW1tb25pYSBFcXVpbGlicml1bSBSZWFjdG9yIEZlZWQsMCwwCkFtbW9uaWEgRXF1aWxpYnJpdW0gUmVhY3RvciBQcm9kdWN0LDAsMApIeWRyb2dlbiBTTVIsMTcxOTQzMy44LDAKTGlxdWlkIEltcHVyZSBBbW1vbmlhLDAsMApOaXRyb2dlbiwxNzU4NDIwNC41LDAKQWxrYW5lIEdhcyBNaXh0dXJlLDAsMApBbW1vbmlhIEdhcyBNaXh0dXJlLDAsMApBaXIgTWl4dHVyZSwwLDAKQW1tb25pYSBPdXRwdXQgR2FzIE1peHR1cmUsMCwwCkFtbW9uaWEgQnlwcm9kdWN0IDIsMCwwCkFtbW9uaWEgQnlwcm9kdWN0IDMsMCwwCkFtbW9uaWEgV2FzdGUsMCwwCkFjZXRhbWlub3BoZW4sMzQ1OTk4MS40MzEsMjM1NjI4MDAKUGFyYSBhbWlub3BoZW5vbCw3NDA5My45NSwwCkFQQVAgV2FzdGUgMSwwLDAKQVBBUCBXYXN0ZSAyLDAsMApCZW56ZW5lLDM3OTM4NDUwLjYsMApDb2tlLDExMjgzODM3NTQsMApCZW56ZW5lIE5hcGh0aGEsNzMzNDkzNzMwNSwwCk5hcGh0aGFsZW5lLDAsMApCZW56ZW5lIEJ5cHJvZHVjdCwwLDAKQmVuemVuZSBXYXN0ZSwwLDAKQWlyLDAsMApDYXJib24gRGlveGlkZSwxMTA4NjczMzcuOCwwCkh5ZHJvZ2VuIEJ5cHJvZHVjdCwwLDAKSHlkcm9nZW4gV2FzdGUsMCwwCklQQSBHYXMgTWl4dHVyZSwwLDAKU3VscGh1cmljIGFjaWQsMjYwNzczMzE2LjcsMApQcm9wYW5lLDEwMzA2MDU3NjA3LDAKUHJvcGVuZSw2NzY5MzU5ODUuNCwwCklQQSBCeXByb2R1Y3QsMCwwCklQQSBXYXN0ZSwwLDAKRXRoYW5lLDQyODQ4NTMyMDYsMApNZXRoYW5lIEZlZWRnYXMsNTkwOTg4ODk5MjAsMApNZXRoYW5lIEdhcyBNaXh0dXJlLDAsMApNZXRoYW5lIExlYW4gR2FzLDAsMApNZXRoYW5lIExlYW4gVEVHLDAsMApNZXRoYW5lIFJpY2ggVEVHLDAsMApNZXRoYW5lIEJ5cHJvZHVjdCAxLDAsMApNZXRoYW5lIEJ5cHJvZHVjdCAyLDAsMApNZXRoYW5lIFdhc3RlLDAsMApNZXRoYW5vbCBCeXByb2R1Y3QgMSwwLDAKTWV0aGFub2wgTml0cm9nZW5fV2F0ZXIgLDAsMApNZXRoYW5vbCBCeXByb2R1Y3QgMiwwLDAKTWV0aGFub2wgV2FzdGUsMCwwCk1ldGhhbm9sIE1lT0hfRXRPSF9XYXRlciwwLDAKTWV0aGFub2wgR2FzIE1peHR1cmUsMCwwCk5pdHJpYyBBY2lkLDMyNjM4NTY1LjcxLDAKTml0cm8gYmVuemVuZSwxMDc1MzcwLjc2OSwwCk5pdHJvYmVuemVuZSBXYXN0ZSwwLDAKTml0cmljIEFjaWQgR2FzIE1peHR1cmUgMSwwLDAKTml0cmljIEFjaWQgV2F0ZXIgTWl4dHVyZSwwLDAKTml0cmljIEFjaWQgR2FzIE1peHR1cmUgMiwwLDAKTml0cmljIEFjaWQgR2FzIE1peHR1cmUgMywwLDAKTml0cmljIEFjaWQgQnlwcm9kdWN0LDAsMApOaXRyaWMgQWNpZCBXYXN0ZSwwLDAKQW5pbGluZSwxNTA2NTY0LjczMywwClBBUCBXYXN0ZSAxLDAsMApQQVAgV2FzdGUgMiwwLDAKUEFQIFdhc3RlIDMsMCwwClBBUCBXYXN0ZSA0LDAsMApDcnVkZSBBR08sMCwwCkJ1dGFuZSw4MjQzNDYzODQ5LDAKRGllc2VsLDIwOTg0MzkzLjI3LDAKQ3J1ZGUgR2FzIE1peHR1cmUsMCwwCkNydWRlIEhOYXBodGhhLDAsMApLZXJvc2VuZSw0MTg1NjEwLjEyNywwCk1peGNydWRlLDY0MDA0ODY4ODU1LDAKTmFwaHRoYSw3MzM0OTM3MzA1LDAKQ3J1ZGUgQnlwcm9kdWN0LDAsMApQZW50YW5lLDQ3MTM3NDYwLjMyLDAKQ3J1ZGUgV2FzdGUsMCwwClByb3BlbmUgQnlwcm9kdWN0LDAsMApQcm9wZW5lIEMySDZfSDIsMCwwClByb3BlbmUgQ0g0X0gyLDAsMApQcm9wZW5lIFdhc3RlLDAsMApTdWxmdXJpYyBBY2lkIEJ5cHJvZHVjdCwwLDAKU3VsZnVyLDEyOTc0MzIyMTUsMApTdWxmdXJpYyBBY2lkIFdhc3RlLDAsMAo=",
  "ri":   "Q29tbW9kaXR5LFdhdGVyLE94eWdlbixOaXRyb2dlbixBaXIsQXJnb24sU3VscGh1cixIZWxpdW0sVG90YWxfTlJfdXNlLFByb2R1Y3Rfb3V0cHV0LFJlc291cmNlX0ludGVuc2l0eSxEb21pbmFudF9yZXNvdXJjZSxCYXNpcyxTb3VyY2UKQWNldGFtaW5vcGhlbiwxNjcyODAwMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDE2NzI4MDAwLjAsMTUzMTUyOTAuMCwxLjA5MjI0MTgwNTQxMTQ1NDksV2F0ZXIsQ29tcHV0ZWQgZnJvbSBQU1QvUFVULCJUaGlzIHN0dWR5IChQU1QgKyBQVVQsIHNjYWxlZCBrZy95cikiCkFjZXRpYyBBY2lkLDEyMzczMDI5MTIwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMTIzNzMwMjkxMjAuMCw0OTQ5MjExNjQ4LjAsMi41LFdhdGVyLExpdGVyYXR1cmUgZXN0aW1hdGUsIkxpdC4gZXN0aW1hdGUgfjIuNSBrZyB3YXRlci9rZyAocHJvY2VzcyB3YXRlciwgbWV0aGFub2wgY2FyYm9ueWxhdGlvbik7IFdCQ1NEIENoZW1pY2FsIFNlY3RvciBMQ0EgTWV0cmljcyAoMjAxNCk7IGVjb2ludmVudCBjaGVtaWNhbHMgTENJLiIKQWNldGljIEFuaHlkcmlkZSwxMzk4MDA4OTQwMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDEzOTgwMDg5NDAwLjAsNDY2MDAyOTgwMC4wLDMuMCxXYXRlcixMaXRlcmF0dXJlIGVzdGltYXRlLExpdC4gZXN0aW1hdGUgfjMuMCBrZyB3YXRlci9rZyAoa2V0ZW5lL2FjZXRpYy1hY2lkIHJvdXRlKTsgV0JDU0QgQ2hlbWljYWwgU2VjdG9yIExDQSBNZXRyaWNzICgyMDE0KS4KQWNldG9uZSwxMzIxODU2LjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMTMyMTg1Ni4wLDg3NjE0NDI1Ni4wLDAuMDAxNTA4NzE5NTg2OTI2MTA1NSxXYXRlcixDb21wdXRlZCBmcm9tIFBTVC9QVVQsIlRoaXMgc3R1ZHkgKFBTVCArIFBVVCwgc2NhbGVkIGtnL3lyKSIKQW1tb25pYSwwLjAsMC4wLDE2NDY1NjU5MDU2LjAwMDAwMiwwLjAsMjU3NTg4MDMyLjAsMC4wLDAuMCwxNjcyMzI0NzA4OC4wMDAwMDIsMTkyOTY2OTI5NDQuMDAwMDA0LDAuODY2NjM3OTg0ODg4NDg0NSxOaXRyb2dlbixDb21wdXRlZCBmcm9tIFBTVC9QVVQsIlRoaXMgc3R1ZHkgKFBTVCArIFBVVCwgc2NhbGVkIGtnL3lyKSIKQmVuemVuZSwyMzcwODg0MTU1Mi4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDIzNzA4ODQxNTUyLjAsMTU4MDU4OTQzNjguMCwxLjUsV2F0ZXIsTGl0ZXJhdHVyZSBlc3RpbWF0ZSxMaXQuIGVzdGltYXRlIH4xLjUga2cgd2F0ZXIva2cgKGFyb21hdGljcyByZWZvcm1pbmcvZXh0cmFjdGlvbik7IERPRS9PU1RJIHdhdGVyIExDSSBmb3IgdHJhbnNwb3J0IGZ1ZWxzICgyMDE1KTsgZWNvaW52ZW50LgpIeWRyb2dlbiBTTVIsMTgxMjkyOTUyOTg0LjAsMC4wLDAuMCwyMTgwMjE0OTM2NDguMCwwLjAsMC4wLDAuMCwzOTkzMTQ0NDY2MzIuMCwzMDUxNzUwODQ4NjQuMCwxLjMwODQ3NjU2NDU1OTQ5OCxBaXIsQ29tcHV0ZWQgZnJvbSBQU1QvUFVULCJUaGlzIHN0dWR5IChQU1QgKyBQVVQsIHNjYWxlZCBrZy95cikiCklzbyBwcm9wYW5vbCwyNzQ3NDE2MzkyLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMjc0NzQxNjM5Mi4wLDIwNjA5Mjc0MDAwLjAsMC4xMzMzMDk3MTI1MTA5NzkyOCxXYXRlcixDb21wdXRlZCBmcm9tIFBTVC9QVVQsIlRoaXMgc3R1ZHkgKFBTVCArIFBVVCwgc2NhbGVkIGtnL3lyKSIKTWV0aGFuZSw1MjQ3OTY3MjcxMC40LDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDUyNDc5NjcyNzEwLjQsMjYyMzk4MzYzNTUyLjAsMC4yLFdhdGVyLExpdGVyYXR1cmUgZXN0aW1hdGUsTGl0LiBlc3RpbWF0ZSB+MC4yIGtnIHdhdGVyL2tnIChuYXR1cmFsLWdhcyBwcm9jZXNzaW5nL2RlaHlkcmF0aW9uKTsgRE9FL09TVEkgd2F0ZXIgTENJICgyMDE1KS4KTWV0aGFub2wsNTU3NjEzMDIyNC4wLDIwMDAzNzYuMCwyNDc1MzM4ODguMDAwMDAwMDMsNTkxNTAyODYxMDQuMCw4NjgyODAuMCwwLjAsODcwMjQuMCw2NDk3NjkwNTg5Ni4wLDEwMjAyNTc1Mzc2LjAsNi4zNjg2NzY4NzgyNzYwNzIsQWlyLENvbXB1dGVkIGZyb20gUFNUL1BVVCwiVGhpcyBzdHVkeSAoUFNUICsgUFVULCBzY2FsZWQga2cveXIpIgpOYXBodGhhLDU0ODk2Nzg5NzYuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCw1NDg5Njc4OTc2LjAsMzE0ODEwMzg1MzQ0LjAsMC4wMTc0MzgwNDkxNjA5MzAwMzQsV2F0ZXIsQ29tcHV0ZWQgZnJvbSBQU1QvUFVULCJUaGlzIHN0dWR5IChQU1QgKyBQVVQsIHNjYWxlZCBrZy95cikiCk5pdHJpYyBBY2lkLDQ3NDgxMjc0ODQ4MC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDQ3NDgxMjc0ODQ4MC4wLDUyOTQ3MDkzODg4MC4wLDAuODk2NzY4MjkwMDMwMDA3MSxXYXRlcixDb21wdXRlZCBmcm9tIFBTVC9QVVQsIlRoaXMgc3R1ZHkgKFBTVCArIFBVVCwgc2NhbGVkIGtnL3lyKSIKTml0cm8gYmVuemVuZSw4MzU3MjgwMDAuMCwwLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCw4MzU3MjgwMDAuMCwxODg1ODM5Nzc2LjAsMC40NDMxNTk1OTk1Nzc3NzQ1LFdhdGVyLENvbXB1dGVkIGZyb20gUFNUL1BVVCwiVGhpcyBzdHVkeSAoUFNUICsgUFVULCBzY2FsZWQga2cveXIpIgpQYXJhIGFtaW5vcGhlbm9sLDEzMzk5NDcyLjAsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMTMzOTk0NzIuMCwxNDAzOTc2MC4wLDAuOTU0Mzk0NjYyMDE3MDE0NSxXYXRlcixDb21wdXRlZCBmcm9tIFBTVC9QVVQsIlRoaXMgc3R1ZHkgKFBTVCArIFBVVCwgc2NhbGVkIGtnL3lyKSIKUHJvcGVuZSwyODExMTIzMDEwMDguMDAwMDYsMC4wLDAuMCwwLjAsMC4wLDAuMCwwLjAsMjgxMTEyMzAxMDA4LjAwMDA2LDEyMzU1NzY1ODQzNjgwLjAsMC4wMjI3NTE1MDc2NDE0MTMzOSxXYXRlcixDb21wdXRlZCBmcm9tIFBTVC9QVVQsIlRoaXMgc3R1ZHkgKFBTVCArIFBVVCwgc2NhbGVkIGtnL3lyKSIKU3VscGh1cmljIGFjaWQsNDM2NzI3NTU0NTYuMCwwLjAsMC4wLDAuMCwwLjAsOTY2NTcyODc3Ni4wLDAuMCw1MzMzODQ4NDIzMi4wLDY5MjA3ODY5ODA4LjAsMC43NzA2OTk2OTYxNDY5MDI3LFdhdGVyLENvbXB1dGVkIGZyb20gUFNUL1BVVCwiVGhpcyBzdHVkeSAoUFNUICsgUFVULCBzY2FsZWQga2cveXIpIgo=",
}
def _sample(key): return base64.b64decode(_B64[key])
RI_BYTES = _sample("ri")


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


# Source-name -> PIOT industry name for Domestic Extraction files that use
# abbreviations / alternative spellings.
DE_ALIASES = {
    "apap": "Acetaminophen", "pap": "Para aminophenol", "ipa": "Iso propanol",
    "hydrogen": "Hydrogen SMR", "hydrogen smr": "Hydrogen SMR",
    "nitrobenzene": "Nitro benzene", "sulfuric acid": "Sulphuric acid",
    "sulphuric acid": "Sulphuric acid", "crude refining": "Naphtha",
}


def load_resource_intensity(source, industries) -> pd.Series:
    """
    Load a per-commodity resource-intensity table (kg natural resource per kg
    product). Accepts a column named like 'Resource_Intensity'/'RI'/'intensity'
    keyed by commodity in the first column (or index). Aligned to industries by
    exact / alias / case-insensitive / substring; unmatched -> 0.
    """
    raw = pd.read_csv(source)
    raw.columns = [str(c).strip() for c in raw.columns]
    name_col = raw.columns[0]
    val_col = None
    for c in raw.columns[1:]:
        cl = c.lower()
        if "resource_intensity" in cl or cl in ("ri", "intensity") or \
           ("resource" in cl and "intensit" in cl):
            val_col = c
            break
    if val_col is None:                     # fall back to a single-value 2nd column
        val_col = raw.columns[1] if len(raw.columns) >= 2 else None
    name_val = {}
    if val_col is not None:
        for _, row in raw.iterrows():
            name_val[str(row[name_col]).strip()] = pd.to_numeric(row[val_col], errors="coerce")
    lower_names = {k.lower(): k for k in name_val}
    out = {}
    for ind in industries:
        v = None
        if ind in name_val:
            v = name_val[ind]
        elif ind.lower() in lower_names:
            v = name_val[lower_names[ind.lower()]]
        else:
            for src in name_val:
                if DE_ALIASES.get(src.strip().lower()) == ind:
                    v = name_val[src]
                    break
            if v is None:
                hits = [n for n in name_val
                        if ind.lower() in n.lower() or n.lower() in ind.lower()]
                if len(hits) == 1:
                    v = name_val[hits[0]]
        out[ind] = float(v) if (v is not None and v == v) else 0.0
    return pd.Series(out, index=industries)


def load_de(source, industries) -> pd.Series:
    """
    Load a Domestic Extraction table and align it to the PIOT industries.

    Accepts either:
      * long format  : columns [Industry/Commodity, DE]
      * wide format  : a row labelled 'DE' across industry-name columns
    Names are matched by exact / alias (DE_ALIASES) / case-insensitive /
    unique-substring. Anything unmatched is 0.
    """
    raw = pd.read_csv(source)
    raw.columns = [str(c).strip() for c in raw.columns]
    first = raw.columns[0]

    name_val: dict[str, float] = {}
    de_row = None
    for _, row in raw.iterrows():
        if str(row[first]).strip().lower() in ("de", "domestic extraction",
                                               "domestic extraction (de)"):
            de_row = row
            break
    if de_row is not None:                                   # wide format
        for c in raw.columns[1:]:
            name_val[c.strip()] = pd.to_numeric(de_row[c], errors="coerce")
    else:                                                    # long format
        valcol = None
        for c in raw.columns[1:]:
            if "de" in c.lower() or "extract" in c.lower():
                valcol = c
                break
        if valcol is None and len(raw.columns) >= 2:
            valcol = raw.columns[1]
        for _, row in raw.iterrows():
            name_val[str(row[first]).strip()] = pd.to_numeric(row[valcol], errors="coerce")

    lower_names = {k.lower(): k for k in name_val}
    result = {}
    for ind in industries:
        val = None
        if ind in name_val:
            val = name_val[ind]
        elif ind.lower() in lower_names:
            val = name_val[lower_names[ind.lower()]]
        else:
            for src in name_val:                             # source -> PIOT alias
                if DE_ALIASES.get(src.strip().lower()) == ind:
                    val = name_val[src]
                    break
            if val is None:
                hits = [n for n in name_val
                        if ind.lower() in n.lower() or n.lower() in ind.lower()]
                if len(hits) == 1:
                    val = name_val[hits[0]]
        result[ind] = float(val) if (val is not None and val == val) else 0.0
    return pd.Series(result, index=industries)


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
# Physical Trade Balance — standalone, PIOT-independent, generic for any network
# --------------------------------------------------------------------------- #
def compute_ptb(imports_bytes: bytes, exports_bytes: bytes) -> pd.DataFrame:
    """
    Physical Trade Balance from the Imports and Exports files ONLY.

        PTB_c = Imports_c - Exports_c

    computed per commodity over the union of commodities appearing in the two
    files. This is fully independent of the PIOT and works for any network.

    Returns a DataFrame indexed by commodity with columns
    Imports (kg/yr), Exports (kg/yr), PTB (kg/yr), sorted by PTB descending.
    """
    impL = read_trade_csv(io.BytesIO(imports_bytes), ["import"])
    expL = read_trade_csv(io.BytesIO(exports_bytes), ["export"])
    commodities = sorted(set(impL.index) | set(expL.index))
    imp = pd.Series({c: float(impL.get(c, 0.0)) for c in commodities})
    exp = pd.Series({c: float(expL.get(c, 0.0)) for c in commodities})
    ptb = (imp - exp).astype(float)
    out = pd.DataFrame({
        "Imports (kg/yr)": imp, "Exports (kg/yr)": exp, "PTB (kg/yr)": ptb,
    })
    out.index.name = "Commodity"
    return out.sort_values("PTB (kg/yr)", ascending=False)


# --------------------------------------------------------------------------- #
# PIOT-based indicators (Physical Trade Balance is NOT part of these)
# --------------------------------------------------------------------------- #
def compute_indicators(piot_bytes: bytes, ri_bytes: bytes | None = None) -> dict:
    """
    Compute the PIOT-based indicators for ANY network:

      Direct   : CSID, SMIR, WGI, PWPR, MUE, MIU   (from the PIOT accounting)
      Leontief : RF, WM, BL, URS                   (from L = (I - A)^-1)

    Physical Trade Balance is computed separately (see compute_ptb) and is not
    included here. Industries are auto-detected from the PIOT, so the function
    is generic for any network.

    The Leontief resource-intensity vector r is taken from the resource-intensity
    file where it matches a commodity; any unmatched commodity falls back to the
    PIOT-derived primary intensity (imports + rest-of-economy inputs) / output,
    so the calculation always produces values for an arbitrary network.

    Returns dict: results (DataFrame [industry x 10 indicators]), accounting,
    industries, spectral_radius, r_source.
    """
    df, industries = load_piot(io.BytesIO(piot_bytes))
    acc = build_accounting(df, industries)
    TMI, PROD, waste = acc["TMI"], acc["PROD"], acc["waste"]
    imports, cross_sector_in, II_in = acc["imports"], acc["cross_sector_in"], acc["II_in"]

    # ----- Direct indicators ------------------------------------------------ #
    CSID = _safe_div(cross_sector_in, TMI, industries) * 100
    SMIR = _safe_div(II_in, TMI, industries) * 100
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

    # PIOT-derived generic resource intensity (primary inputs per unit output).
    r_piot = (imports.to_numpy() + acc["ROE_in"].to_numpy()) * inv_x
    r_source = "PIOT-derived (imports + rest-of-economy) / output"
    if ri_bytes is not None:
        RI = load_resource_intensity(io.BytesIO(ri_bytes), industries).to_numpy(dtype=float)
        matched = RI > 0
        if matched.any():
            r = np.where(matched, RI, r_piot)        # RI where available, else PIOT fallback
            r_source = ("resource-intensity file"
                        + ("" if matched.all() else " (PIOT-derived fallback for unmatched commodities)"))
        else:
            r = r_piot
    else:
        r = r_piot

    w = waste.to_numpy() * inv_x                                     # waste intensity
    y = (acc["final_demand"] + acc["exports_col"] + acc["ROE_out"]).to_numpy(dtype=float)

    PRM = pd.Series(r @ L, index=industries)                        # internal (not reported)
    BL = pd.Series(L.sum(axis=0), index=industries)
    WM = pd.Series(w @ L, index=industries)
    RF = PRM * pd.Series(y, index=industries)
    URS = _safe_div(PRM - pd.Series(r, index=industries), PRM, industries) * 100

    results = pd.DataFrame({
        "CSID": CSID, "SMIR": SMIR, "WGI": WGI, "PWPR": PWPR, "MUE": MUE, "MIU": MIU,
        "RF": RF, "WM": WM, "BL": BL, "URS": URS,
    })
    results.index.name = "Industry"

    accounting = pd.DataFrame({
        "II_in": acc["II_in"], "Self_use": acc["self_use"],
        "Cross_sector_in": cross_sector_in, "ROE_in": acc["ROE_in"],
        "Imports": imports, "TMI": TMI, "II_out": acc["II_out"],
        "ROE_out": acc["ROE_out"], "Exports": acc["exports_col"],
        "Final_demand": acc["final_demand"], "PROD": PROD,
        "Waste": waste, "Total_output": acc["TO"],
    })

    return dict(results=results, accounting=accounting, industries=industries,
                spectral_radius=float(np.max(np.abs(np.linalg.eigvals(A)))),
                r_source=r_source)


# --------------------------------------------------------------------------- #
# Indicator metadata: formula, unit, description, reference, decision question
# --------------------------------------------------------------------------- #
# direction: +1  -> a HIGH value means HIGH priority/need for the decision
#            -1  -> a LOW value means HIGH priority/need for the decision
# Physical Trade Balance is a standalone indicator computed from the Imports and
# Exports files only (see compute_ptb) — independent of the PIOT indicators below.
PTB_META = dict(
    key="PTB", name="Physical Trade Balance", unit="kg/yr", diverging=True, direction=+1,
    formula=r"\mathrm{PTB}_c = \mathrm{IMP}_c - \mathrm{EXP}_c",
    description="The net physical trade position of each commodity — imports minus exports, "
                "in kilograms per year, taken straight from the trade files. A positive value "
                "marks a net importer (the network leans on foreign supply for that commodity); "
                "a negative value marks a net exporter (domestic output exceeds domestic use). "
                "It pinpoints where supply security rides on trade flows that could be "
                "interrupted, and is computed independently of the input–output table.",
    reference="Eurostat (2018), Economy-wide material flow accounts handbook.",
    decision="Secure domestic supply / onshore?")

DIRECT_INDICATORS = [
    dict(key="CSID", name="Cross-Sector Input Dependency", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{CSID}_j = \dfrac{\sum_{i\neq j} Z_{ij}}{\mathrm{TMI}_j}\times 100",
         description="The share of an industry's total material input supplied by OTHER "
                     "modelled industries, excluding its own recycled (diagonal) flow. It "
                     "measures how tightly a commodity is coupled to its domestic upstream "
                     "partners: a high value signals strong internal interdependence, so a "
                     "disruption in one sector propagates readily into this one.",
         reference="Miller & Blair (2009), Input-Output Analysis.",
         decision="Strengthen cross-sector supply coordination?"),
    dict(key="SMIR", name="Secondary Material Input Rate", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{SMIR}_j = \dfrac{\sum_{i} Z_{ij}}{\mathrm{TMI}_j}\times 100",
         description="The share of total material input met by intermediate materials "
                     "circulating within the modelled system, including the industry's own "
                     "recycled (self-use) flow. It is a proxy for how much feedstock is "
                     "'secondary' — already inside the industrial network rather than freshly "
                     "extracted — so higher values point to greater circularity potential. "
                     "Because it combines cross-sector and self-use flows it is not, on its "
                     "own, a strict circularity metric.",
         reference="Eurostat (2018); Haas et al. (2015), J. Ind. Ecol.",
         decision="Leverage secondary-material / circular sourcing?"),
    dict(key="WGI", name="Waste Generation Intensity", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{WGI}_j = \dfrac{W_j}{\mathrm{TMI}_j}\times 100",
         description="The percentage of an industry's total material input that leaves as "
                     "waste rather than product. It captures how much of everything entering "
                     "the process is lost — combining conversion inefficiency and unusable "
                     "co-streams — so the highest values flag the strongest candidates for "
                     "closed-loop recovery and waste valorisation.",
         reference="Eurostat (2018); Nakamura & Kondo (2009), Waste Input-Output.",
         decision="Need for closed-loop recovery?"),
    dict(key="PWPR", name="Physical Waste-to-Product Ratio", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{PWPR}_j = \dfrac{W_j}{\mathrm{PROD}_j}",
         description="Kilograms of waste generated per kilogram of saleable product — the "
                     "waste-to-product ratio at the factory gate. Unlike WGI (normalised by "
                     "input), this expresses waste relative to useful output, so a value of 2 "
                     "means two kilograms of waste accompany every kilogram of product. It "
                     "highlights processes where waste minimisation or redesign would deliver "
                     "the largest absolute reductions.",
         reference="Allwood et al. (2011), Material efficiency.",
         decision="Priority for waste minimisation / process redesign?"),
    dict(key="MUE", name="Material Utilization Efficiency", unit="%", diverging=False, direction=-1,
         formula=r"\mathrm{MUE}_j = \dfrac{\mathrm{PROD}_j}{\mathrm{TMI}_j}\times 100",
         description="The percentage of input mass converted into useful, non-waste product. "
                     "It is the efficiency counterpart of WGI — under strict mass balance "
                     "MUE + WGI ≈ 100% — so a low value means most material entering the "
                     "process is lost and signals a strong need for efficiency improvement.",
         reference="Allwood et al. (2011); OECD (2008).",
         decision="Need for process-efficiency improvement?"),
    dict(key="MIU", name="Material Intensity per Unit Product", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{MIU}_j = \dfrac{\mathrm{TMI}_j}{\mathrm{PROD}_j}",
         description="The kilograms of material input required to make one kilogram of "
                     "product — the reciprocal of material efficiency. It is a direct measure "
                     "of how material-hungry a process is: high values raise feedstock, "
                     "handling and logistics costs and mark the material-intensive stages of "
                     "the supply chain.",
         reference="Schmidt-Bleek (1993), MIPS concept.",
         decision="Reduce material intensity / feedstock burden?"),
]

LEONTIEF_INDICATORS = [
    dict(key="RF", name="Resource Footprint", unit="kg/yr", diverging=False, direction=+1,
         formula=r"\mathrm{RF}_j = \mathrm{PRM}_j \times y_j,\quad \mathrm{PRM}_j=\sum_i r_i\,l_{ij}",
         description="The total primary natural resource embodied in the final demand "
                     "delivered by the sector, found by propagating each unit of demand "
                     "through the whole upstream chain (Leontief inverse) weighted by each "
                     "sector's resource intensity r. It answers 'how much natural resource — "
                     "water, oxygen, nitrogen, sulphur and the like — is ultimately drawn to "
                     "satisfy demand for this commodity', direct plus indirect. Larger "
                     "footprints identify the commodities whose consumption places the "
                     "greatest burden on primary resources.",
         reference="Wiedmann et al. (2015), PNAS; Tukker et al. (2016), Glob. Env. Change.",
         decision="Manage upstream resource burden / footprint?"),
    dict(key="WM", name="Waste Multiplier", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{WM}_j = \sum_i w_i\,l_{ij}",
         description="The total waste generated across the entire upstream supply chain per "
                     "unit of a sector's output, obtained by weighting the Leontief inverse "
                     "by each sector's waste intensity. It captures embodied (indirect) waste "
                     "that gate-level indicators miss, so a high value means producing this "
                     "commodity triggers large waste generation elsewhere in the network.",
         reference="Nakamura & Kondo (2002), J. Ind. Ecol.; Duchin (1990).",
         decision="Target upstream (embodied) waste reduction?"),
    dict(key="BL", name="Backward Linkage", unit="kg/kg", diverging=False, direction=+1,
         formula=r"\mathrm{BL}_j = \sum_i l_{ij}",
         description="The column sum of the Leontief inverse: the total output pulled from "
                     "the whole network — directly and indirectly — for each unit of final "
                     "demand for the commodity. It measures how deep and wide a commodity's "
                     "upstream dependence runs; a value well above 1 marks a commodity whose "
                     "demand strongly stimulates, and depends on, the rest of the supply "
                     "chain — a key resilience consideration.",
         reference="Rasmussen (1956); Hirschman (1958), Strategy of Econ. Development.",
         decision="Prioritise supply-chain resilience (deep upstream reliance)?"),
    dict(key="URS", name="Upstream Resource Share", unit="%", diverging=False, direction=+1,
         formula=r"\mathrm{URS}_j = \left(\dfrac{\mathrm{PRM}_j - r_j}{\mathrm{PRM}_j}\right)\times 100",
         description="The share of a commodity's total embodied primary resource that is "
                     "drawn from upstream sectors rather than from its own production stage, "
                     "as a percentage. A high value means most of the resource burden sits "
                     "with suppliers, so securing or decarbonising the feedstock chain "
                     "matters more than acting on the final stage alone.",
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


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _human(num):
    if num == 0: return "0"
    sign = "-" if num < 0 else ""; n = abs(num)
    for div, suf, dec in [(1e9,"B",2),(1e6,"M",2),(1e3,"k",1)]:
        if n >= div: return f"{sign}{n/div:.{dec}f} {suf}"
    return f"{sign}{n:.0f}"

def ptb_symlog_figure(ptb_df, top_n=25):
    s = ptb_df["PTB (kg/yr)"]
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32*len(s)+1.2)))
    colors = ["#c0392b" if v >= 0 else "#3E6E8E" for v in s]
    ax.barh(s.index.astype(str), s.values, color=colors)
    ax.axvline(0, linewidth=0.8, color="black"); ax.set_xscale("symlog", linthresh=1)
    m = float(np.abs(s.values).max()) if len(s) else 0.0
    if m > 0:
        lim = 10 ** math.ceil(math.log10(m)); ax.set_xlim(-lim*4, lim*4)
    ax.invert_yaxis()
    ax.set_xlabel("Imports − Exports (kg/yr)   |   ← net exporter        net importer →")
    ax.grid(axis="x", alpha=0.25)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); return fig

def indicator_bar(series, meta):
    dfc = series.rename("value").reset_index()
    dfc = dfc.rename(columns={dfc.columns[0]: "Industry"})
    col = "#3E6E8E" if meta["key"] in ("CSID","SMIR","WGI","PWPR","MUE","MIU") else "#4a6fa8"
    return (alt.Chart(dfc.dropna(subset=["value"])).mark_bar(color=col).encode(
        x=alt.X("value:Q", title=f"{meta['name']} ({meta['unit']})", axis=alt.Axis(format="~s")),
        y=alt.Y("Industry:N", sort="-x", title=None),
        tooltip=["Industry", alt.Tooltip("value:Q", title=meta["key"], format=",.4g")],
    ).properties(height=26*dfc["value"].notna().sum()+30))

def render_indicator(meta, series, n):
    st.markdown(f"#### {n}. {meta['name']} ({meta['key']})")
    st.latex(meta["formula"]); st.markdown(meta["description"])
    st.caption(f"Reference: {meta['reference']}")
    st.altair_chart(indicator_bar(series, meta), use_container_width=True)
    st.divider()



# --------------------------------------------------------------------------- #
# UI — one page
# --------------------------------------------------------------------------- #
st.title("EW-MFA Indicators & Decision Support")
st.markdown("Material-flow indicators for a manufacturing network. **Physical Trade Balance** "
    "is computed from the Imports/Exports files alone; the **other indicators** come from the "
    "PIOT. Both are generic for any network.")

_mfa = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mfa_overview.png")
if os.path.exists(_mfa):
    with st.container(border=True):
        st.markdown("**What material-flow analysis (MFA) delivers in manufacturing**")
        lc, mc, rc = st.columns([1,6,1]); mc.image(_mfa, use_container_width=True)
        st.caption("Reference: Eurostat, Economy-wide material flow accounts (EW-MFA).")

st.header("1 · Physical Trade Balance")
st.markdown("Computed **only from the Imports and Exports files** — it does not use the PIOT "
            "and is independent of every other indicator.")
st.latex(r"\mathrm{PTB}_c = \mathrm{IMP}_c - \mathrm{EXP}_c")

with st.container(border=True):
    c1, c2 = st.columns(2)
    up_imp = c1.file_uploader("Imports CSV", type=["csv"], key="imp")
    up_exp = c2.file_uploader("Exports CSV", type=["csv"], key="exp")
    ptb_sample = st.checkbox("Use the bundled sample trade files", value=True, key="ptb_sample")
    if st.button("Compute Physical Trade Balance", type="primary"):
        try:
            if ptb_sample and up_imp is None and up_exp is None:
                ib, eb, lbl = _sample("imp"), _sample("exp"), "bundled sample"
            elif up_imp is not None and up_exp is not None:
                ib, eb, lbl = up_imp.getvalue(), up_exp.getvalue(), "your uploaded files"
            else:
                st.error("Upload BOTH an Imports and an Exports CSV (or tick the sample box)."); st.stop()
            st.session_state["ptb"] = compute_ptb(ib, eb); st.session_state["ptb_src"] = lbl
            st.success(f"Physical Trade Balance computed from {lbl}.")
        except Exception as exc:
            st.error(f"Could not compute PTB: {exc}")

if "ptb" in st.session_state:
    ptb = st.session_state["ptb"]
    st.caption(f"Source: {st.session_state.get('ptb_src','')} · {len(ptb)} commodities")
    n_show = st.slider("Show top N commodities by |PTB|", 5, min(60, len(ptb)), min(25, len(ptb)))
    st.pyplot(ptb_symlog_figure(ptb, n_show))
    st.caption("Red = net importer (PTB > 0) · Blue = net exporter (PTB < 0). Symmetric-log axis.")
    st.dataframe(ptb.style.format({"Imports (kg/yr)":"{:,.0f}","Exports (kg/yr)":"{:,.0f}","PTB (kg/yr)":"{:,.0f}"}),
                 use_container_width=True)
    st.download_button("Download PTB (CSV)", ptb.to_csv().encode(), "physical_trade_balance.csv", "text/csv")
else:
    st.info("No Physical Trade Balance yet — press the button above.", icon="⚖️")

st.divider()

st.header("2 · EW-MFA Indicators (from the PIOT)")
st.markdown("Upload the PIOT to compute the direct indicators (CSID, SMIR, WGI, PWPR, MUE, MIU) "
            "and the Leontief upstream indicators (RF, WM, BL, URS). Industries are auto-detected.")

with st.container(border=True):
    up_piot = st.file_uploader("PIOT CSV", type=["csv"], key="piot")
    ind_sample = st.checkbox("Use the bundled sample PIOT", value=True, key="ind_sample")
    if st.button("Compute the Indicators", type="primary"):
        try:
            if ind_sample and up_piot is None:
                pb, lbl = _sample("piot"), "bundled sample"
            elif up_piot is not None:
                pb, lbl = up_piot.getvalue(), "your uploaded PIOT"
            else:
                st.error("Upload a PIOT CSV (or tick the sample box)."); st.stop()
            st.session_state["computed"] = compute_indicators(pb, RI_BYTES)
            st.session_state["ind_src"] = lbl
            st.success(f"Indicators computed from {lbl}.")
        except Exception as exc:
            st.error(f"Computation failed: {exc}")

if "computed" in st.session_state:
    out = st.session_state["computed"]; res = out["results"]
    st.caption(f"Source: {st.session_state.get('ind_src','')} · {len(out['industries'])} industries "
               f"· spectral radius of A = {out['spectral_radius']:.4f}")
    tab_d, tab_l, tab_dec = st.tabs(["Direct indicators", "Leontief indicators", "Decision heatmap"])
    with tab_d:
        for i, meta in enumerate(DIRECT_INDICATORS, 1):
            render_indicator(meta, res[meta["key"]], i)
        st.download_button("Download all indicator values (CSV)", res.to_csv().encode(),
                           "ewmfa_indicators.csv", "text/csv")
    with tab_l:
        st.caption(f"Resource-intensity source: {out.get('r_source','')}")
        st.latex(r"A = Z\,\hat{x}^{-1}, \qquad L = (I-A)^{-1} = I + A + A^{2} + \cdots")
        for i, meta in enumerate(LEONTIEF_INDICATORS, 1):
            render_indicator(meta, res[meta["key"]], i)
    with tab_dec:
        st.markdown("Each indicator ranked into **Low / Medium / High** priority (within-network "
                    "percentiles, oriented so High = most action needed). Physical Trade Balance is not included.")
        labels, _ = decision_matrix(res)
        order = [m["key"] for m in ALL_INDICATORS]
        row_label = {m["key"]: f"{m['key']} — {m['decision']}" for m in ALL_INDICATORS}
        rows = []
        for k in order:
            for ind in res.index:
                rows.append({"Decision": row_label[k], "Industry": ind,
                             "Level": str(labels.loc[ind, k]), "Value": res.loc[ind, k]})
        long = pd.DataFrame(rows); row_sort = [row_label[k] for k in order]
        base = alt.Chart(long)
        heat = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
            x=alt.X("Industry:N", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Decision:N", sort=row_sort, title=None, axis=alt.Axis(labelLimit=360)),
            color=alt.Color("Level:N", scale=alt.Scale(domain=["Low","Medium","High"],
                range=[LEVEL_COLORS["Low"],LEVEL_COLORS["Medium"],LEVEL_COLORS["High"]]),
                legend=alt.Legend(title="Priority", orient="top")),
            tooltip=["Industry","Decision",alt.Tooltip("Value:Q",format=",.4g"),"Level"])
        txt = base.mark_text(baseline="middle", fontSize=9, color="white", fontWeight="bold").encode(
            x="Industry:N", y=alt.Y("Decision:N", sort=row_sort), text="Level:N")
        st.altair_chart((heat+txt).properties(height=44*len(order)+20), use_container_width=True)
        dd = pd.DataFrame([{"Indicator": f"{m['name']} ({m['key']})", "Decision question": m["decision"],
                            "High priority when": ("value is HIGH" if m["direction"]>0 else "value is LOW")}
                           for m in ALL_INDICATORS])
        st.dataframe(dd, use_container_width=True, hide_index=True)
        st.download_button("Download decision matrix (CSV)", labels.to_csv().encode(),
                           "ewmfa_decision_matrix.csv", "text/csv")
else:
    st.info("No indicators yet — upload a PIOT (or tick the sample box) and press Compute.", icon="\U0001F9EE")
