# CII & EEOI Voyage Calculator

A Streamlit app that estimates a vessel voyage's attained CII (AER), CII rating (A-E),
and EEOI from vessel particulars, a speed/fuel-consumption curve, and per-leg voyage data.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, sign in, and click "New app".
3. Select this repo/branch and set the main file path to `app.py`.
4. Deploy — Streamlit Cloud will host it at a public `*.streamlit.app` URL and
   redeploy automatically on every push to the connected branch.

## Notes

- CII reference-line parameters and rating boundaries follow IMO MEPC.352(78).
- Reduction factors (Z) for 2023-2026 follow MEPC.354(78). Factors for 2027-2030
  are not yet formally adopted by IMO; this app uses a placeholder linear
  extrapolation, clearly flagged in the app when selected.
- For estimation/planning purposes only — not a substitute for verified IMO DCS/SEEMP reporting.
