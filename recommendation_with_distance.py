import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import pairwise_distances
import random
from numpy import inf
from numpy import linalg as LA



dataset = pd.read_csv("data/features_msd_lda_sp.csv")
dataset.drop(["Unnamed: 0"], 1, inplace=True)
dataset.columns

genres = dataset.genre.unique()
dataset.genre.replace(genres, np.arange(10), inplace=True)
dataset

node_id = dataset['id']
feature_set = dataset.drop(['track_id','artist_name','title','id'], 1)


for column in feature_set.columns:
    max_of_column = feature_set[column].max()
    feature_set[column] = feature_set[column].apply(lambda x: x / max_of_column)

distance_matrix = pairwise_distances(feature_set.loc[:],feature_set.loc[:],metric = 'l2')
real_distance_matrix = distance_matrix.copy()
distance_matrix = 1/distance_matrix 
distance_matrix[distance_matrix == inf ] = 0
total = distance_matrix.sum(axis=1)

current = 0
def get_distribution(current):
    dists = distance_matrix[current]
    start = 0
    result = {}
    for s in range(len(dists)):
        if dists[s] != current:
            d_s = dists[s] / total[current]
            result[s] = (start, start + d_s)
            start += d_s

        if start >= 1:
            break
    
    return result



seed_songs = feature_set.ix[np.random.choice(feature_set.index, 1)]
seed_songs



def get_next_song(current,seed_songs):
    while current in seed_songs.index:
        distrib = get_distribution(current)
        r = random.random()
        for k, v in distrib.items():
            if v[0] <= r < v[1]:
                current = k
                break
    return current

avg_seed_dist = 0
avg_playlist_dist = 0
NUMBER_OF_TRIAL = 10
for index in range(0, NUMBER_OF_TRIAL + 1):
    get_recommended_songs = []
    recommendation_count = 10
    seed_song_index = seed_songs.index
    seed_song = seed_song_index[0]
    while recommendation_count > 0 : 
        next_song = get_next_song(seed_song,seed_songs)
        while next_song  in get_recommended_songs:
            next_song = get_next_song(seed_song,seed_songs)

        recommendation_count = recommendation_count -1
        get_recommended_songs.append(next_song)
    seedSong = dataset.loc[seed_song]['artist_name'], dataset.loc[seed_song]['title'] ,dataset.loc[seed_song]['id'] 
    print("Seed song")
    print(seedSong)
    print("------")
    recommended_songs = [(dataset.loc[i]['artist_name'], dataset.loc[i]['title'], dataset.loc[i]['id']) for i in get_recommended_songs]
    print("Playlist")
    print(dataset.loc[get_recommended_songs]['artist_name'] + " by " + dataset.loc[get_recommended_songs]['title'] + " id: " + dataset.loc[get_recommended_songs]['id'])
    seed_distance = []
    for i in get_recommended_songs:
        #print('Distance between artist name: ' +  dataset.loc[seed_song]['artist_name'] + ', title: ' + dataset.loc[seed_song]['title'] + ' and artist name: ' + dataset.loc[i]['artist_name'] + ', title: ' + dataset.loc[i]['title'])
        dist = LA.norm(feature_set.loc[seed_song].get_values() - feature_set.loc[i].get_values())
        seed_distance.append(dist)

    avg_seed_dist += np.mean(seed_distance)
    distance = []
    for i in range(len(get_recommended_songs)-1):
        first_song = get_recommended_songs[i]
        second_song = get_recommended_songs[i+1]    
        #print('Distance between artist name: ' +  dataset.loc[first_song]['artist_name'] + ', title: ' + dataset.loc[first_song]['title'] + ' and artist name: ' + dataset.loc[second_song]['artist_name'] + ', title: ' + dataset.loc[second_song]['title'])
        dist = LA.norm(feature_set.loc[seed_song].get_values() - feature_set.loc[i].get_values())
        distance.append(dist)

    avg_playlist_dist += np.mean(distance)   


max_distance = np.amax(real_distance_matrix)  
max_distance
avg_seed_dist / (NUMBER_OF_TRIAL * max_distance)
avg_playlist_dist/ (NUMBER_OF_TRIAL * max_distance)
np.mean(real_distance_matrix)/max_distance

