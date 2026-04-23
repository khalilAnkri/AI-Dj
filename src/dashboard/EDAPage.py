import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from importDataset import importBigQuerry
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI-DJ Hit Predictor - EDA")

st.title("AI-DJ Hit Predictor - EDA")

df = importBigQuerry()

df["hit"] = (df["popularity"] >= 65).astype(int)
df = df.drop(columns=["popularity"])
df = df.drop(columns=["Unnamed: 0"])

explicit_col = df['explicit'].astype(int)

df = df.select_dtypes(include=["number"])

page = st.sidebar.selectbox("Select analysis", [
    "Dataset description",
    "Correlation matrix",
    "Pair plot",
    "PCA 2D",
    "PCA 3D"
])

if page == "Dataset description":
    st.image("spotify.jpg")
    st.write("Dataset description")
    st.write(df.describe(include='all'))
    
elif page == "Correlation matrix":
    st.write("Correlation matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=False, ax=ax)
    st.pyplot(fig)

elif page == "Pair plot":
    st.write("Pair plot")
    pairPlot = sns.pairplot(df, kind='scatter', corner=True)
    st.pyplot(pairPlot)

elif page == "PCA 2D":
    st.write("PCA 2D")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    selectionPCA = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'duration_ms']

    data = df[selectionPCA]

    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(data)

    pca2D = PCA(n_components=2)
    data2D = pca2D.fit_transform(dataScaled)

    
    sc = axes[0, 0].scatter(data2D[:, 0], data2D[:, 1], c=df['key'].astype('category').cat.codes, s=7, alpha=0.5)
    axes[0, 0].set_title("PCA - Category = key")
    axes[0, 0].set_xlabel("PCA1")
    axes[0, 0].set_ylabel("PCA2")
    
    plt.colorbar(sc, ax=axes[0, 0])

    sc = axes[0, 1].scatter(data2D[:, 0], data2D[:, 1], c=df['mode'].astype(
        'category').cat.codes, s=7, alpha=0.5)
    axes[0, 1].set_title("PCA - Category = mode")
    axes[0, 1].set_xlabel("PCA1")
    axes[0, 1].set_ylabel("PCA2")

    plt.colorbar(sc, ax=axes[0, 1])

    sc = axes[1, 0].scatter(data2D[:, 0], data2D[:, 1], c=df['time_signature'].astype(
        'category').cat.codes, s=7, alpha=0.5)
    axes[1, 0].set_title("PCA - Category = time_signature")
    axes[1, 0].set_xlabel("PCA1")
    axes[1, 0].set_ylabel("PCA2")

    plt.colorbar(sc, ax=axes[1, 0])

    sc = axes[1, 1].scatter(data2D[:, 0], data2D[:, 1], c=explicit_col.astype(
        'category').cat.codes, s=7, alpha=0.5)
    axes[1, 1].set_title("PCA - Category = explicit")
    axes[1, 1].set_xlabel("PCA1")
    axes[1, 1].set_ylabel("PCA2")

    plt.colorbar(sc, ax=axes[1, 1])
    st.pyplot(fig)
    
elif page == "PCA 3D":
    st.write("PCA 3D")
    
    selectionPCA = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'duration_ms']

    data = df[selectionPCA]

    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(data)
    
    pca3D = PCA(n_components=3)
    data3D = pca3D.fit_transform(dataScaled)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter(data3D[:, 0], data3D[:, 1], data3D[:, 2])
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    
    st.pyplot(fig)