import streamlit as st
import seaborn as sns
import json
import matplotlib.pyplot as plt

st.title("Vertical Lines Chart of Random Number Frequency")

files = ["first_version.json","second_version.json","third_version.json","fourth_version.json","fifth_version.json","sixth_version.json","seventh_version.json"]
figures=[]
for file in files:
    with open('number_data/'+file,'r') as f:
        data=json.loads(f.read())
        f.close()

    white_balls=[]
    red_balls=[]
    for pick in data.values():
        white,red=pick
        for ball in white:
            white_balls.append(ball)
        red_balls.append(red)
    # penguins = sns.load_dataset("penguins")
    # Set Seaborn color palette
    colors = sns.color_palette("bright")

    # Create Seaborn plot
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(x=sorted(white_balls)).set(title=f'{file} White Balls')
    st.pyplot(fig)

