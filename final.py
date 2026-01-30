import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
pd.set_option('display.max_columns', None)

my_var="""variables


1.   df
2.   df_cat
3.   df_num
4.   df_num_scaled    array
5.   df_scaled
6.   df_pca           array
7.   pca_df
8.   df_final_num
9.   df_final

"""

df=pd.read_csv('dataset.csv')

#df.head()

#df.info()

df["track_id"].duplicated().sum()

df=df.drop_duplicates(subset="track_id")

df.isnull().sum()

df.dropna(subset=["artists", "album_name", "track_name"], inplace=True)

df = df.astype({
    "track_id": "string",
    "artists": "string",
    "album_name": "string",
    "track_name": "string",
    "track_genre": "string"
})

df=df[df["duration_ms"] > 0]

df=df[df["popularity"].between(0, 100)]

df=df[df["tempo"] > 0]

df=df[df["loudness"].between(-60, 0)]

audio_cols = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness",
    "liveness", "valence"
]

for col in audio_cols:
    df=df[df[col].between(0, 1)]

#df.info()

df.to_csv("clean_music_data.csv", index=False)
print("Data cleaning complete")

#df.shape

#df.describe()

df.isnull().sum()

features_num=['danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_ms', 'time_signature']
features_cat=['track_id', 'artists', 'album_name', 'track_name', 'explicit', 'track_genre']
df=df.drop(['Unnamed: 0'], axis=1)

df_cat=df[features_cat]

df["popularity"].mean()

plt.figure(figsize=(25,25))
df.hist()
plt.tight_layout()
plt.show()

plt.hist(df["popularity"], bins=20)
plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.show()

df.groupby("track_genre")["popularity"].mean().head()

df_corr=df[features_num].corr()
plt.figure(figsize=(6, 6))
sns.heatmap(df_corr, cmap='coolwarm')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(df["energy"], df["loudness"], alpha=0.3)
plt.xlabel("Energy")
plt.ylabel("loudness")

plt.figure(figsize=(6, 6))
plt.scatter(df["speechiness"], df["liveness"], alpha=0.3)
plt.xlabel("speechiness")
plt.ylabel("liveness")

plt.figure(figsize=(6, 6))
plt.scatter(df["danceability"], df["valence"], alpha=0.3)
plt.xlabel("danceability")
plt.ylabel("valence")

df_num=df[features_num]
scale=StandardScaler()
df_num_scaled=scale.fit_transform(df_num)

#df_num_scaled

df_scaled=pd.DataFrame(df_num_scaled, columns=df_num.columns)

#df_scaled

n=10
pca=PCA(n_components=n)
df_pca=pca.fit_transform(df_scaled)
#df_pca

plt.figure(figsize=(6,6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.3)

pca_df = pd.DataFrame(
   df_pca,
    columns=[f"pc_{i+1}" for i in range (n)],
)

#pca_df

df_num=df_num.reset_index()

pca_df=pca_df.reset_index()
df_num.drop('index', axis=1, inplace=True)
pca_df.drop('index', axis=1, inplace=True)
#pca_df
#df_final_num=pd.concat([ df_num,pca_df], axis=1)
df_final_num=df_num.join(pca_df)

#df_final_num

#df_final=pd.concat([df_final_num, df_cat], axis=1)
#df_final=df_final_num.join(df_cat, how="inner")
df_final=pd.concat([df_final_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
df_final.head(10)
#df_final.shape
knn = NearestNeighbors(
    n_neighbors=21, metric='cosine'
)

knn.fit(df_pca)
def calc(song_name, topn=10):
    index=df_final[df_final["track_name"]==song_name].index[0]

    distances, indices = knn.kneighbors(
        [df_pca[index]],
        n_neighbors=topn + 1
    )

    distances = distances.flatten()[1:]
    indices = indices.flatten()[1:]
    similarity = 1 - distances

    results = df_final.iloc[indices][
        ['track_name', 'artists', 'album_name']
    ].copy()

    results['similarity'] = similarity
    return results



st.header('Song Recommender')
st.write('Enter Song')
n=st.text_input('Song')
if st.button('run'):
    st.write(calc(n))