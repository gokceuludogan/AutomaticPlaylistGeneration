import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random 
def weighted_norm(arr1, arr2, ind=-1, weight=10):
    if ind != -1:
        arr1[ind] *= weight
        arr2[ind] *= weight
    return np.linalg.norm(arr1-arr2)

def create_input_for_clustering(dataset):
    genres = dataset.genre.unique()
    for index, genre in zip(range(0, len(genres)),genres):
        dataset.loc[dataset['genre'] == genre, 'genre'] = index 
    #dataset.drop(string_features, 1, inplace = True)
    return dataset.as_matrix()   

def clustering_with_model(model, data):
	X =  create_input_for_clustering(data.copy())
	#kmeans = KMeans(n_clusters=319)
	fitted_model = model.fit(X)
	# Getting the cluster labels
	labels = fitted_model.predict(X)
	# Centroid values
	centroids = fitted_model.cluster_centers_
	inertia = fitted_model.inertia_
	print("inertia:", inertia)
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))     

	all_distances = [[weighted_norm(X[ind1], X[ind2]) for ind1 in range(0,X.shape[0])] for ind2 in range(0,X.shape[0])]
	max_distance = np.max(all_distances)

	random_track_indices = random.sample(range(len(X)), 10)
	test_input = create_input_for_clustering(data.copy())[random_track_indices]
	cluster_numbers = kmeans.predict(test_input)
	column_dict = {k: v for v, k in enumerate(list(dataset))}
	seed_rec_distances = []
	rec_to_rec_distances = []
	seed_to_all_distances = []
	for i, (cluster_number, random_track_ind) in enumerate(zip(cluster_numbers, random_track_indices)):
	    index = fitted_model.labels_ == cluster_number
	    indices = [ind for ind, value in zip(range(0, len(labels)), index) if value == True]
	    d = ([weighted_norm(X[ind], test_input[i]) for ind in indices])
	    ind = np.argsort(d)[::-1][:10]
	    indices = [indices[x] for x in ind]
	    print(dataset.iloc[random_track_ind]["title"] + " by " + dataset.iloc[random_track_ind]["artist_name"])
	    print('======')
	    print(dataset.iloc[indices]["title"] + " by " + dataset.iloc[indices]["artist_name"])
	    seed_rec_distances.append(np.mean([all_distances[ind][random_track_ind] for ind in indices]))
	    rec_to_rec_distances.append(np.mean([np.mean([all_distances[ind1][ind2] for ind1 in indices if ind1 != ind2]) for ind2 in indices]))
	    seed_to_all_distances.append(np.mean([all_distances[ind][random_track_ind] for ind in range(0,X.shape[0])]))

	print('mean distance between seed and recommended songs: ' + str(np.mean(seed_rec_distances)/max_distance))
	print('mean distance between all recommended songs: ' + str(np.mean(rec_to_rec_distances)/max_distance))
	print('mean distance between seed and all songs: ' + str(np.mean(seed_to_all_distances)/max_distance))

def main():
	input_data_path = "features_msd_lda_sp.csv"
	NO_OF_CLUSTER = 100
	dataset = pd.read_csv(input_data_path)
	dataset.drop(["Unnamed: 0"], 1, inplace=True)

	string_features = ["track_id", "id", "artist_name", "title"]
	datacopy = dataset.copy()
	datacopy.drop(string_features, 1, inplace = True)
	genres = dataset.genre.unique()
	for index, genre in zip(range(0, len(genres)),genres):
	    datacopy.loc[datacopy['genre'] == genre, 'genre'] = index 
	for column in datacopy.columns:
	    max_of_column = datacopy[column].abs().max()
	    datacopy[column] = datacopy[column].apply(lambda x: x / max_of_column)

	clustering_with_model(KMeans(n_clusters=319), datacopy)
	clustering_with_model(AffinityPropagation(), datacopy)

if __name__ == "__main__":
	main()
