#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tabulate

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
from naive_bayes import GaussianNaiveBayes

import time
import statistics


def load_set():
    """
    Load the spambase dataset
    :return: feature vectors and labels as arrays
    """
    data = []
    labels = []
    with open("spambase.data", "r") as input_file:
        for line in input_file:
            fields = line.split(",")
            data.append([float(fields[i]) for i in range(54)])
            labels.append(int(fields[57]))
    return np.array(data), np.array(labels)


def confusion_matrix(trained_classifier, X_test, y_test, class_names):
    """
    Plot normalized confusion matrix
    """
    np.set_printoptions(precision=2)
    title = "Normalized confusion matrix"
    disp = plot_confusion_matrix(trained_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()


def autotune(X_train, X_test, y_train, y_test, class_names, cv=5):
    """
    Search parameter combinations in order to find a good combination
    """
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree':[2]}]

    # there are many possible choices of objective functions
    scores = ['accuracy']#['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring=score, n_jobs=-1, cv=cv
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        fit_times = clf.cv_results_['mean_fit_time']
        score_times = clf.cv_results_['mean_score_time']

        table = [(mean, std, params, fit_time, score_time)
                 for mean, std, params, fit_time, score_time
                 in zip(means, stds, clf.cv_results_['params'], fit_times, score_times)]
        print(tabulate.tabulate(table,
                                headers=("Mean\nscore","Std\ndev","Parameters", "Mean fitting\ntime [s]",
                                         "Mean evaluation\ntime [s]"),
                                floatfmt=".3f")
              )

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred, target_names=class_names))
        print()


def main():
    """
    Example usage
    """
    class_names = ["ham", "spam"]
    X, y = load_set()

    # split the set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    print("\n---------------------OUR GNB ---------------------\n")
    # create and train a classifier
    start_time = time.perf_counter()
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    fit_time = time.perf_counter()

    # run the classifier on test set
    y_pred = gnb.predict(X_test)
    pred_time = time.perf_counter()

    # print report
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"Fit time: {fit_time-start_time:.2f}s, Prediction time: {pred_time-start_time:.2f}s")
    # very slow compared to scikit's implementation, probably because of the use of dicts - I should be change them to
    # arrays

    print("\n-------------------SCIKIT GNB---------------------\n")
    # should work +- the same as scikit's one
    start_time = time.perf_counter()
    skgnb = GaussianNB()
    skgnb.fit(X_train, y_train)
    fit_time = time.perf_counter()
    y_pred = skgnb.predict(X_test)
    pred_time = time.perf_counter()

    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"Fit time: {fit_time-start_time:.2f}s, Prediction time: {pred_time-start_time:.2f}s")

    print()
    # or some graphs (see https://scikit-learn.org/ for more examples)
    confusion_matrix(gnb,X_test,y_test, class_names)

    print()
    # SVMs have many params, use grid search to tune them, CAREFULL SUPER SLOW -> use smaller training sets for
    # development (internally uses cross-validation
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=200, train_size=500, random_state=0)
    autotune(X_train2, X_test2, y_train2, y_test2, class_names)

    # cross-validation example
    print("10-fold cross-validation scores for GNB")
    crs_val_scrs = cross_val_score(GaussianNB(), X, y, cv=10, n_jobs=-1)
    print(crs_val_scrs)
    print(f"mean: {statistics.mean(crs_val_scrs):.2f}")

if __name__ == '__main__':
    main()
