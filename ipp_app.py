import streamlit as st
import pickle as pkl
import pandas as pd

st.set_page_config(layout = "wide")

st.title('IPL Win Predictor')

teams = pkl.load(open('teams.pkl','rb'))
cities = pkl.load(open('cities.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))

## first roe and column

col1,col2,col3 = st.columns(3)

with col1:
    batting_team = st.selectbox('Select the Batting Team',sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the Bowling Team',sorted(teams))


with col3:
    city = st.selectbox('Select the host city',sorted(cities))



target = st.number_input('Target_Score',min_value=0,max_value=720,step=1)


col4,col5,col6 = st.columns(3)

with col4:
    score = st.number_input('Score',min_value=0,max_value=720,step=1)

with col5:
    overs = st.number_input('Overs Done',min_value=0,max_value=20,step=1)

with col6:
    wickets = st.number_input('Wickets Fell',min_value=0,max_value=10,step=1)


if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 -(overs*6)
    wickets =10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'Score': [score],
    'target_left': [runs_left],
    'Remaining_ball': [balls_left],
    'Wickets': [wickets],
    'crr': [crr],
    'rrr': [rrr],
})


    result = model.predict_proba(input_df)
    win = result[0][1]
    loss = result[0][0]
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")



                           








