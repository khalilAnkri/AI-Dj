# Exploratory Data Analysis

Firstly, our dataset is composed of 114,000 observations and 21 variables. This dataset consists of 114,000 Spotify tracks across 125 genres. Our set of variables are devided into 3 parts. The first part contains the identifier variables such as the track_id, artists, album_name, track_name and track_genre. Then, the second part is related to the target variable, popularity (integer, between 1 and 100), calculated by Spotify's algorithm based on total plays and recency. Further, before the training step, the variable will be transformed into a binary variable called hit between 0 (flop) and 1 (hit) where the threshold is set to popularity >= 65. We con observe that our dataset is made up of 9477 hits (8.31%). Then, the third group of variables which are the exploratory variables are composed of some audio features such as danceability, tempo, duration_ms and so on. 

Afterwards, we can notice that our dataset is composed of just one observation over all our dataset which contain a missing value for artists, album_name and track_name, which is not disturbing because those are related to identifiers. 

Then, by analyzing the correlation pot our variables, we can notice that most of the variables are not very correlated because the two extreme correlation coefficient are 0.76 and -0.73, and the others coefficient are low positive/negative correlation coefficient. This suggests that most features contains different information. Moreover, by performing a PCA with 2 and 3 PC without the identifiers and the non categorical features, we can observe that the three first PC explain a small part of the variance respectivement 0.28, 0.15 and 0.12. This suggests that the data cannot be well summarized into 2 or 3 components meaning that the information is spread across many features as noticed before.

# ML model

