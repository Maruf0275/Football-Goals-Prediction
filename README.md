# Football GoalCast

A Streamlit dashboard that predicts estimated full-time Premier League goals
from a saved Scikit-learn regression model and recent team form.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `football_goalcast.pkl` in the same folder as `app.py`.

## Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Select the `Maruf0275/Football-Goals-Prediction` repository.
3. Select the `main` branch.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

Streamlit Cloud installs the packages from `requirements.txt` automatically.

## Notes

This is a demonstration model. Predictions are estimated mean goals, not
guaranteed final scores or shot-level expected goals (xG). Team form remains
fixed at the date the model was trained.

## License

MIT