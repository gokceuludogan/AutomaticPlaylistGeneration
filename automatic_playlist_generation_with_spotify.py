import spotipy
import requests
import spotipy.util as util
import pprint, json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import random


CLIENT_ID = 'CLIENT_ID'
CLIENT_SECRET = 'CLIENT_SECRET'
scope = 'user-library-read playlist-modify user-read-private'
username = 'USERNAME'
PLAYLIST_NO = 0
feature_list = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "id"]


def get_token(username, scope, client_id, client_secret):
    return util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri='http://localhost/')

def read_user_saved_tracks(sp):
    limit_of_track = 50
    offset_of_track = 0
    get_track = True
    audio_features = np.array(feature_list)
    while get_track:
        results = sp.current_user_saved_tracks(limit=limit_of_track, offset=offset_of_track)
        len_of_tracks = 0
        for item in results['items']:
            len_of_tracks += 1
            track = item['track']
            #print(track['id'] + " - " + track['name'])
            features_in_json = get_audio_features(sp, [track['id']])
            track_feature = [value for key, value in features_in_json[0].items()  if key in feature_list]
            audio_features = np.vstack([audio_features, track_feature])
        offset_of_track += len_of_tracks
        if len_of_tracks != 50:
            get_track = False
            print(offset_of_track)
    print("Total saved tracks " + str(offset_of_track))
    return audio_features

def get_audio_features(sp, track_id_list):
    return sp.audio_features(tracks=track_id_list)

def create_playlist_for_user(sp, username, playlist_name):
    return sp.user_playlist_create(username, playlist_name)["id"]

def add_tracks_to_playlist(sp, track_id_list, playlist_id):
    return sp.user_playlist_add_tracks(username, playlist_id, track_ids)

def weighted_norm(arr1, arr2, ind=-1, weight=10):
    if ind != -1:
        arr1[ind] *= weight
        arr2[ind] *= weight
    return np.linalg.norm(arr1-arr2)

def get_user_saved_tracks()
    token = get_token(username, scope, CLIENT_ID, CLIENT_SECRET)
    sp = spotipy.Spotify(auth=token)
    return read_user_saved_tracks(sp)

def load_classifier(path):
    classifier = None
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        classifier = pickle.load(f)
    return classifier

def main():
    audio_features = get_user_saved_tracks()
    user_saved_tracks_df = pd.DataFrame(data = audio_features[1:, 0:], columns=audio_features[0,0:])
    audio_features = audio_features[:, :-1]
    audio_feature_df = pd.DataFrame(data = audio_features[1:, 0:], columns=audio_features[0,0:])
    audio_features = audio_feature_df.as_matrix()
    kmeans_user = KMeans(n_clusters=15)
    kmeans_user = kmeans_user.fit(audio_features)
    # Getting the cluster labels
    labels = kmeans_user.predict(audio_features)
    classifier = load_classifier("classifier")    

    random_track_indices = random.sample(range(len(X)), 10)
    NUMBER_OF_PLAYLIST_GENERATED = 1
    random_track_indices = random.sample(range(audio_features.shape[0]), NUMBER_OF_PLAYLIST_GENERATED)
    #random_track_indices = [208]
    test_input = audio_feature_df.as_matrix()[random_track_indices]
    nones = np.zeros((test_input.shape[0], X.shape[1]-len(feature_list)+1))
    test_input = np.append(nones, test_input[:,1:], axis=1)
    genres = classifier.predict(test_input)
    genres = [x/np.max(genres) for x in genres]
    audio_copy = audio_feature_df.copy()
    for ind, genre in zip(random_track_indices, genres):
        audio_copy.iloc[ind][0] = genre
    test_input = audio_copy.as_matrix()[random_track_indices]
    cluster_numbers = kmeans_user.predict(test_input)
    for i, (cluster_number, random_track_ind) in enumerate(zip(cluster_numbers, random_track_indices)):
        index = kmeans_user.labels_ == cluster_number
        indices = [ind for ind, value in zip(range(0, len(labels)), index) if value == True]
        d = ([weighted_norm(audio_copy.iloc[ind], test_input[i]) for ind in indices])
        ind = np.argsort(d)[::-1][:10]
        indices = [indices[x] for x in ind]
        print(user_saved_tracks_df.iloc[random_track_ind]["id"])
        print ('======')
        track_ids = user_saved_tracks_df.iloc[indices]["id"].values.tolist()
        sp.trace = False
        print(track_ids)
        playlist_name = "automatic-generated-playlist-" + str(PLAYLIST_NO)
        PLAYLIST_NO += 1
        playlists = sp.user_playlist_create(username, playlist_name)
        playlist_id = playlists["id"]
        results = sp.user_playlist_add_tracks(username, playlist_id, track_ids)

if __name__ == "__main__":
    main()
