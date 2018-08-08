import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import spotipy
import requests
import spotipy.util as util
import pprint, json
import numpy as np

CLIENT_ID = 'SPOTIFY_CLIENT_ID'
CLIENT_SECRET = 'SPOTIFY_CLIENT_SECRET'
scope = 'user-library-read playlist-modify user-read-private'

def get_dataset(path):
	return pd.read_csv(path)

def get_song_names_and_counts_of_data(path):
	file = open(path, 'r')
	lines = [x for x in file.readlines() if not x.startswith('#')]
	top_words = lines[0][1:].split(',')
	splitted_lines = [x.split(',') for x in lines[1:]]
	song_word_counts = {x[0]:{top_words[int(y.split(':')[0])-1]:int(y.split(':')[1]) for y in x[2:]} for x in splitted_lines[1:]}
	song_names = [x[0] for x in splitted_lines]
	song_name_df = pd.DataFrame(song_names, columns=['track_id'])
	word_count_df = pd.DataFrame(song_word_counts)
	word_count_df.fillna(0, inplace=True)
	word_count_df = word_count_df.transpose()
	word_count_df.index.name = 'track_id'
	word_count_df.reset_index(inplace = True)
	return song_name_df, word_count_df


def  get_track_id_by_name_and_artist(token, track_name, artist_name):
	response = requests.get('https://api.spotify.com/v1/search',
					headers={ 'authorization': "Bearer " + token}, 
					params={ 'q': 'track:' + track_name +  ' artist:' + artist_name, 'type': 'track' })
	if response.ok == False:
		return -2        
	tracks = json.loads(response.text)["tracks"]
	if tracks["total"] == 0:
		return -1
	return tracks["items"][0]["id"]

def get_token(username, scope, client_id, client_secret):
	return util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri='http://localhost/')

def get_audio_features(sp, track_id_list):
	return sp.audio_features(tracks=track_id_list)

def get_audio_features_given_list(features, feature_list):
	audio_features  = np.array(feature_list)
	not_found = 0
	for index, row in features.iterrows():
		if index % 1000 == 0:
			token = get_token(username, scope, CLIENT_ID, CLIENT_SECRET)
			sp = spotipy.Spotify(auth=token)
		track_id_msd = row.track_id
		track_name = row.title
		artist_name = row.artist_name
		#print("Index\t" + str(index) + "\tTrack\t" + track_name + "\tArtist\t" + artist_name)
		track_id = get_track_id_by_name_and_artist(token, track_name, artist_name)
		if track_id == -2:
			token = get_token(username, scope, CLIENT_ID, CLIENT_SECRET)
			track_id = get_track_id_by_name_and_artist(token, track_name, artist_name)
		if track_id != -1:
			features_in_json = get_audio_features(sp, [track_id])
			track_feature = [value for key, value in features_in_json[0].items()  if key in feature_list[1:]]
			audio_features = np.vstack([audio_features, [track_id_msd] +track_feature])
		else: 
			print("Track not found "+ track_name + "\t" +artist_name)
			not_found += 1
	return audio_features

def get_text_features(msd_data_path, mxm_data_path, text_featured_output):
	songs_df = get_dataset(msd_data_path)
	song_name_df, word_count_df = get_song_names_and_counts_of_data(mxm_data_path)
	msd_with_mxm = pd.merge(songs_df, song_name_df, on='track_id')
	msd_with_word_counts = pd.merge(msd_with_mxm[['track_id']], word_count_df, on='track_id')
	msd_with_word_counts.set_index('track_id', inplace = True)
	lda = LatentDirichletAllocation()
	lyrics_categories = lda.fit_transform(msd_with_word_counts)
	lda_feature_df = pd.DataFrame(lyrics_categories)
	features = pd.merge(msd_with_mxm, lda_feature_df, left_index=True, right_index=True)
	features.to_csv(text_featured_output)
	return features

def get_all_audio_features(spotify_username, audio_featured_output):

	token = get_token(spotify_username, scope, CLIENT_ID, CLIENT_SECRET)
	sp = spotipy.Spotify(auth=token)
	feature_list = ["track_id_msd", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "id"]
	audio_features = get_audio_features_given_list(features, feature_list)
	audio_features_frame = pd.DataFrame(data = audio_features[1:, 0:], columns=audio_features[0,0:])
	audio_features_frame = audio_features_frame.rename(index=str, columns={"track_id_msd": "track_id"})
	audio_features_frame.to_csv(audio_featured_output)
	return audio_features_frame


if __name__ == "__main__":
	features = get_text_features('data/msd_genre_dataset.csv', 'data/mxm_dataset_train.txt', 'data/msd_extra_features.csv')
	username = 'SPOTIFY_USERNAME'
	audio_features_frame = get_all_audio_features(username, 'data/audio_features.csv')
	features_with_audio = pd.merge(features, audio_features_frame, on='track_id')
	#features_with_audio.drop(["Unnamed: 0"], 1, inplace = True)
	features_with_audio.to_csv("data/features_msd_lda_sp.csv")
