import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
%matplotlib inline
import matplotlib.pyplot as plt

def create_input_and_label_for_genre(dataset, genre):
    label = np.where(dataset['genre']==genre, 1, 0)
    feature = dataset.loc[:, dataset.columns != 'genre']
    feature.drop(string_features, 1, inplace = True)
    return feature.as_matrix(), label

def get_SVC():
    return SVC()

def get_LinearSVC():
    return LinearSVC()

def get_logistic_regression():
    return LogisticRegression()        

def get_random_forest_classifier(n_features):
    return RandomForestClassifier(max_depth=n_features, random_state=0)

def create_input_and_label_for_multiclass(dataset, genres):
    label = dataset[['genre']]
    for index, genre in zip(range(0, len(genres)),genres):
        label.loc[dataset['genre'] == genre, 'genre'] = index 
    feature = dataset.loc[:, dataset.columns != 'genre']
    feature.drop(string_features, 1, inplace = True)
    return feature.as_matrix(), label['genre']    

def classification_with_different_classifiers(X_train, X_test, y_train, y_test, clf, clf_name):
#     global train_acc_list
#     global test_acc_list
#     global cv_acc_list
    clf.fit(X_train, y_train)
    #print("Feature importance", clf.feature_importances_)
    cv = 10
    cross_validation_scores = cross_val_score(clf, X, y, cv=cv)
    #print("Cross validation scores", cross_validation_scores)
    mean_cvs = sum(cross_validation_scores)/len(cross_validation_scores)
    predictions = clf.predict(X_test)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, predictions)
    if clf_name in train_acc_list:
        train_acc_list[clf_name].append(train_acc)
    else:
        train_acc_list[clf_name] = [train_acc]
    if clf_name in test_acc_list:
        test_acc_list[clf_name].append(test_acc)
    else:
        test_acc_list[clf_name] = [test_acc]
    if clf_name in cv_acc_list:
        cv_acc_list[clf_name].append(mean_cvs)
    else:
        cv_acc_list[clf_name] = [mean_cvs]
    print("Train Accuracy :: ", train_acc)
    print("Cross validation Accuraccy :: ", mean_cvs)
    print("Test Accuracy  :: ", test_acc)

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(predictions, label, predict_names, label_names):
    mat = confusion_matrix(predictions, label)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=predict_names,
            yticklabels=label_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    return plt

def classify(X_train, y_train, X_test, y_test, classifier, classifier_name):
    cv = 10
    classifier.fit(X_train, y_train)
    cross_validation_scores = cross_val_score(classifier, X, y, cv=cv)
    mean_cvs = sum(cross_validation_scores)/len(cross_validation_scores)
    predictions = classifier.predict(X_test)
    train_acc = accuracy_score(y_train, classifier.predict(X_train))
    test_acc = accuracy_score(y_test, predictions)
    print("Results for classifier\t: " + classifier_name)
    print("Train Accuracy\t\t\t:: ", train_acc)
    print("Cross validation Accuracy\t:: ", mean_cvs)
    print("Test Accuracy\t\t\t:: ", test_acc)
    conf_mat = plot_confusion_matrix(predictions, y_test, genres, genres)
    conf_mat.show()
def main():
    input_data_path = "features_msd_lda_sp.csv"
    dataset = pd.read_csv(input_data_path)
    dataset.drop(["Unnamed: 0"], 1, inplace=True)
    genres = dataset.genre.unique()
    dataset.genre.value_counts()
    string_features = ["track_id", "id", "artist_name", "title"]
    classifier_names = ["Random Forest multiclass", "Random Forest One Vs All", "Logistic Regression multiclass", "Logistic Regression One Vs All", "SVM", "Linear SVM"]

    train_acc_listtrain_ac  = {}
    test_acc_list = {}
    cv_acc_list = {}
    for genre in genres:
        X, y = create_input_and_label_for_genre(dataset.copy(), genre)
        #onevsallRF = OneVsRestClassifier(RandomForestClassifier(max_depth=X.shape[1], random_state=0))
        RF = get_random_forest_classifier(X.shape[1])
        LR = get_logistic_regression()
        svc = get_SVC()
        linearSVC = get_LinearSVC()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        print("Binary classification based on genre: "+ genre)
        print()
        for classifier, classifier_name in zip([RF, LR, svc, linearSVC], ["Random Forest", "Logistic Regression", "SVC", "Linear SVC"]):
            print ('Results for ' + classifier_name)
            classification_with_different_classifiers(X_train, X_test, y_train, y_test, classifier, classifier_name)
            print ('-------------------------')

    for classifier_name in ["Random Forest", "Logistic Regression", "SVC", "Linear SVC"]:
        overall_train_acc = sum(train_acc_list[classifier_name])/len(train_acc_list[classifier_name])
        overall_test_acc = sum(test_acc_list[classifier_name])/len(test_acc_list[classifier_name])
        overall_cv_acc = sum(cv_acc_list[classifier_name])/len(cv_acc_list[classifier_name])
        print("Overall results for " + classifier_name)    
        print("Train Accuracy :: ", overall_train_acc)
        print("Test Accuracy  :: ", overall_test_acc)
        print("CV Accuracy    :: ", overall_cv_acc)        

    X, y = create_input_and_label_for_multiclass(dataset.copy(), genres)

    onevsallRF = OneVsRestClassifier(RandomForestClassifier(max_depth=X.shape[1], random_state=0))
    multiclassRF = get_random_forest_classifier(X.shape[1])
    onevsallLR = OneVsRestClassifier(LogisticRegression())
    multiclassLR = get_logistic_regression()
    svc = get_SVC()
    linearSVC = get_LinearSVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #cross_val_score_multiclass = cross_val_score(multiclass X, y, cv=cv)
    #cross_val_score_onevsall = cross_val_score(onevsall, X, y, cv=cv)
    for classifier, classifier_name in zip([onevsallRF, multiclassRF, onevsallLR, multiclassLR, svc, linearSVC], classifier_names):
        classify(X_train, y_train, X_test, y_test, classifier, classifier_name)

    X,y = create_input_and_label_for_multiclass(dataset.copy(), genres)
    multiclassLR = get_logistic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    multiclassLR.fit(X_train, y_train)
    with open('classifier', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(multiclassLR, f, pickle.HIGHEST_PROTOCOL)        

if __name__ == "__main__":
    main()        