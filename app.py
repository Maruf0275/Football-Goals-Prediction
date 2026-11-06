"""Streamlit dashboard for the Football GoalCast model."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'football_goalcast.pkl'

st.set_page_config(page_title='Football GoalCast', page_icon='⚽', layout='wide')
st.title('Football GoalCast')
st.caption('Pre-match Premier League goal predictions using Scikit-learn')

@st.cache_resource
def read_model():
    with MODEL_PATH.open('rb') as file:
        return pickle.load(file)  # Load only a pickle file you created and trust.


def mean_recent(rows, key, fallback, venue=None):
    if venue:
        rows = [entry for entry in rows if entry['venue'] == venue]
    rows = rows[-5:]
    return float(np.mean([entry[key] for entry in rows])) if rows else fallback


def make_features(home, away, history):
    hh = history.get(home, [])
    ah = history.get(away, [])
    return {
        'HomeTeam': home,
        'AwayTeam': away,
        'home_scored5': mean_recent(hh, 'scored', 1.35),
        'home_conceded5': mean_recent(hh, 'conceded', 1.35),
        'home_points5': mean_recent(hh, 'points', 1.3),
        'home_home_scored5': mean_recent(hh, 'scored', 1.35, 'home'),
        'home_home_conceded5': mean_recent(hh, 'conceded', 1.35, 'home'),
        'away_scored5': mean_recent(ah, 'scored', 1.35),
        'away_conceded5': mean_recent(ah, 'conceded', 1.35),
        'away_points5': mean_recent(ah, 'points', 1.3),
        'away_away_scored5': mean_recent(ah, 'scored', 1.35, 'away'),
        'away_away_conceded5': mean_recent(ah, 'conceded', 1.35, 'away'),
        'home_history_count': min(len(hh), 5),
        'away_history_count': min(len(ah), 5),
    }

try:
    saved = read_model()
except FileNotFoundError:
    st.error('Missing football_goalcast.pkl. Add the model file beside app.py and redeploy the app.')
    st.stop()

st.write(f"**Selected regression model:** {saved['model_name']}")
st.write(f"**Historical match data through:** {saved['data_through']}")
teams = saved['teams']

left, right = st.columns(2)
with left:
    home = st.selectbox('Home team', teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 0)
with right:
    away = st.selectbox('Away team', teams, index=teams.index('Chelsea') if 'Chelsea' in teams else 1)

if home == away:
    st.warning('Choose two different teams.')
elif st.button('Predict full-time goals', type='primary'):
    inputs = pd.DataFrame([make_features(home, away, saved['history'])])[saved['features']]
    predicted = np.maximum(saved['model'].predict(inputs)[0], 0.0)
    col_home, col_away = st.columns(2)
    col_home.metric(f'{home} estimated goals', f'{predicted[0]:.2f}')
    col_away.metric(f'{away} estimated goals', f'{predicted[1]:.2f}')
    st.caption('Predicted mean goals are not guaranteed final scores or shot-level expected goals (xG).')
    st.subheader('Pre-match information used')
    st.dataframe(inputs.T.rename(columns={0: 'Value'}), use_container_width=True)

st.info('Demonstration only. Team form remains frozen at the date on which you last trained the model.')
