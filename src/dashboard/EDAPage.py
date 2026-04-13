import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from google.cloud import bigquery
from importDataset import importBigQuerry

df = importBigQuerry()

df["hit"] = (df["popularity"] >= 65).astype(int)
df = df.drop(columns=["popularity"])
df = df.drop(columns=["Unnamed: 0"])

df = df.select_dtypes(include=["number"])

st.title("AI-DJ Hit Predictor")

st.image("spotify.jpg")

theme = st.sidebar.selectbox("Select theme", ["Dark", "Light"])

if theme == "Dark":
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
else:
    sns.set_style("whitegrid")
    plt.style.use("default")

st.write()