import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from google.cloud import bigquery

BUCKET = "https://storage.googleapis.com/ai-dj-487610-streamlit-data"

client = bigquery.Client(project="ai-dj-487610")

housing = client.query("SELECT * FROM `ai-dj-487610.housing.housing`").to_dataframe()
iris    = client.query("SELECT * FROM `ai-dj-487610.iris.iris`").to_dataframe()
stocks  = client.query("SELECT * FROM `ai-dj-487610.all_stocks_5yr.all_stocks_5yr`").to_dataframe()
wine    = client.query("SELECT * FROM `ai-dj-487610.winequality_red.winequality_red`").to_dataframe()

dataset = st.sidebar.selectbox("Select dataset", [
    "California House Price",
    "S&P 500 stock data",
    "Iris dataset",
    "Wine Quality",
])
theme = st.sidebar.selectbox("Select theme", ["Dark", "Light"])

if theme == "Dark":
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
else:
    sns.set_style("whitegrid")
    plt.style.use("default")

color_map = {
    "NEAR BAY":   "#a6cee3",
    "INLAND":     "#b2df8a",
    "NEAR OCEAN": "#fb9a99",
    "ISLAND":     "#fdbf6f",
    "<1H OCEAN":  "#cab2d6",
}
scaler = MinMaxScaler(feature_range=(1, 100))
housing["size"]  = scaler.fit_transform(housing[["median_house_value"]])
housing["color"] = housing["ocean_proximity"].map(color_map)

if dataset == "California House Price":
    st.write(housing)
    fig, ax = plt.subplots()
    sns.histplot(housing["median_house_value"], bins=50, ax=ax, edgecolor="black")
    ax.set_xlabel("Median House Value")
    ax.set_ylabel("Number of Houses")
    ax.set_title("Histogram of Median House Values")
    st.pyplot(fig)
    st.map(housing, latitude="latitude", longitude="longitude",
           size="size", color="color")

elif dataset == "S&P 500 stock data":
    stock_symbol = st.selectbox("Select a stock symbol", stocks["Name"].unique())
    stock_data   = stocks[stocks["Name"] == stock_symbol]
    st.write(stock_data)
    st.line_chart(stock_data["close"])
    st.scatter_chart(stock_data[["close", "volume"]])

elif dataset == "Iris dataset":
    st.write(iris)
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(iris.select_dtypes(include="number").corr(), cmap="YlGnBu", annot=True, ax=ax)
    ax.set_title("Correlation matrix of iris")
    st.pyplot(fig)
    st.subheader("Pair plot of iris")
    st.pyplot(sns.pairplot(iris, kind='scatter', corner=True))
    st.subheader("Violin plot")
    fig, ax = plt.subplots()
    sns.violinplot(data=iris, x="Species", y="SepalLengthCm", ax=ax)
    ax.set_title("Violin plot of iris")
    st.pyplot(fig)

elif dataset == "Wine Quality":
    st.write(wine)
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(wine.select_dtypes(include="number").corr(), cmap="YlGnBu", annot=False, ax=ax)
    ax.set_title("Correlation matrix of wine")
    st.pyplot(fig)
    st.subheader("Pair plot of wine")
    st.pyplot(sns.pairplot(wine, kind='scatter', corner=True))
    st.subheader("Violin plot")
    fig, ax = plt.subplots()
    sns.violinplot(data=wine, x="quality", y="alcohol", ax=ax)
    ax.set_title("Violin plot of wine")
    st.pyplot(fig)