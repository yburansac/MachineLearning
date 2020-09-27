#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tabulate
import pickle
import scipy.stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize, minmax_scale, Normalizer, MinMaxScaler, MaxAbsScaler, binarize
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from naive_bayes import GaussianNaiveBayes

import time
import statistics



def load_set():
    """
    Load the spambase dataset
    :return: feature vectors and labels as numpy arrays
    """
    data = []
    labels = []
    with open("spambase.data", "r") as input_file:
        for line in input_file:
            fields = line.split(",")
            data.append([float(fields[i]) for i in range(54)])
            labels.append(int(fields[57]))
    return np.array(data), np.array(labels)

def select_svm_hyperparameters(test_set, test_set_labels):
    Cs = [10.0 ** x for x in range(-2,3)]
    gammas = [10.0 ** x for x in range(-8, 1)] + ["scale"]

    tuned_parameters = {
        "rbf": {'kernel': ['rbf'], 'gamma': gammas, 'C': [10.0 ** x for x in range(-1,10)]},
        "linear": {'kernel': ['linear'], 'C': Cs},
        "poly d2": {'kernel': ['poly'], 'C': Cs, 'coef0': [0, 1], 'degree': [2]},
        "poly d3": {'kernel': ['poly'], 'C': Cs, 'coef0': [0, 1], 'degree': [3]}
    }

    # there are many possible choices of objective functions
    score = 'accuracy'  # ['precision', 'recall']

    results = {
        "rbf": {
            "scores":[],
            "fit_times":[],
            "score_times":[]
        },
        "linear": {
            "scores":[],
            "fit_times":[],
            "score_times":[]
        },
        "poly d2": {
            "scores":[],
            "fit_times":[],
            "score_times":[]
        },
        "poly d3": {
            "scores":[],
            "fit_times":[],
            "score_times":[]
        }
    }

    verbosity = 1
    cv_folds = 10

    for group in tuned_parameters.keys():
        clf = GridSearchCV(
            SVC(max_iter=1000000),
            tuned_parameters[group],
            scoring=score, n_jobs=-1, cv=cv_folds, verbose=verbosity, refit=False
        )
        clf.fit(test_set, test_set_labels)

        scores = clf.cv_results_['mean_test_score']
        fit_times = clf.cv_results_['mean_fit_time']
        score_times = clf.cv_results_['mean_score_time']

        results[group]["scores"].extend(scores)
        results[group]["fit_times"].extend(fit_times)
        results[group]["score_times"].extend(score_times)
        results[group]["param_sets"] = clf.cv_results_['params'].copy()
        results[group]["best"] = clf.best_params_
        results[group]["best_score"] = clf.best_score_

    return results

def print_table_of_2_params(results,param_x,param_y, title="Some table"):
    results=results['estimator'][0].cv_results_
    param_list_x = set()
    param_list_y = set()
    for param_set in results['params']:
        param_list_x.add(param_set[param_x])
        param_list_y.add(param_set[param_y])
    from numbers import Number
    param_list_x = sorted(list(param_list_x), key=lambda x: (isinstance(x, Number), x))
    param_list_y = sorted(list(param_list_y), key=lambda x: (isinstance(x, Number), x))
    table = [[None for col in range(len(param_list_y))] for row in range(len(param_list_x))]
    for i in range(len(results["params"])):
        C, gamma = results["params"][i][param_x], results["params"][i][param_y]
        table[param_list_x.index(C)][param_list_y.index(gamma)]=100*results["mean_test_score"][i]
    print_table_title(title)
    print(tabulate.tabulate(table,headers=param_list_y, showindex=param_list_x))

def print_results(results, dataset_names, pipeline_names, group_names, metric,
                  scope='gridsearch',  apply_filter=None, title="Some table", as_percentage=False,
                  calculate_statistic=statistics.mean, floatfmt=".2f"):

    index = []
    for group_name in group_names:
        if scope == 'gridsearch':
            params=results[dataset_names[0]][pipeline_names[0]][group_name]['estimator'][0].cv_results_['params']
        else:
            params=[group_name]
        index.extend(params)
    table = [[] for i in range(len(index))]
    header = []
    for dataset_name in dataset_names:
        for pipeline_name in pipeline_names:
            j = 0
            header.append(f'{dataset_name}\n{pipeline_name}')
            for group_name in group_names:
                if scope == "gridsearch":
                    metric_scores = results[dataset_name][pipeline_name][group_name]['estimator'][0].cv_results_[metric]
                    for i in range(len(metric_scores)):
                        table[j].append(metric_scores[i] * 100 if as_percentage else metric_scores[i])
                        j += 1
                else:
                    metric_scores = calculate_statistic(results[dataset_name][pipeline_name][group_name][metric])
                    table[j].append(metric_scores * 100 if as_percentage else metric_scores)
                    j += 1

    print_table_title(title)

    if apply_filter:
        index, table = zip(*filter(apply_filter, zip(index, table)))

    print(tabulate.tabulate(table, headers=header, showindex=index, floatfmt=floatfmt))


def print_table_title(title, terminal_width=120):
    print(terminal_width * '-')
    centering_offset = (terminal_width - len(title))//2
    print(centering_offset * " " + title)
    print(terminal_width * '-')

def transform_tf_to_tf_idf(X):
    X_ = np.empty_like(X)
    df = np.zeros(X.shape[1])
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            if X[row,col] > 0:
                df[col] += 1
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            X_[row,col] = X[row,col]*(-np.log((df[col]+1)/(len(X)+1)))
    return X_

class TFtoTF_IDF_trnasformer(TransformerMixin):
    def _init__(self):
        self.idf=None
    def fit(self, X,y=None):
        df = np.count_nonzero(X, 0)
        smoothing = 1
        n = X.shape[0] + smoothing
        df += smoothing
        self.idf = -np.log(df/n)
        return self
    def transform(self,X):
        return self.idf * X

def plot_learning_curves_macro(estimator1, estimator2, e1_name, e2_name, title, datset, dataset_labels, e1_is_slow=False, e2_is_slow=False):
    from learning_curves import plot_learning_curve, plot_comarison_of_learing_curves
    cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) if e1_is_slow else ShuffleSplit(n_splits=100,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    cv2 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) if e2_is_slow else ShuffleSplit(n_splits=100,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    train_sizes = np.geomspace(0.01,1.0,15)
    if estimator2:
        plot_comarison_of_learing_curves(estimator1, estimator2, e1_name, e2_name, title, datset, dataset_labels, cv1=cv1, cv2=cv2, n_jobs=-1, train_sizes=train_sizes)
    else:
        plot_learning_curve(estimator1, title, datset, dataset_labels, cv=cv1, n_jobs=-1, train_sizes=train_sizes)
    plt.show()

def main():
    from pathlib import Path
    class_names = ["ham", "spam"]

    X, y = load_set()
    #X, _, y, _ = train_test_split(X,y,train_size=0.12)

    recompute = False

    pickled_train_set_results = "gridsearch.pickle"

    if Path(pickled_train_set_results).is_file() and not recompute:
        with open(pickled_train_set_results, 'rb') as file:
            results = pickle.load(file)
    else:
        pipelines = {
            "plain": Pipeline([
                ("estimator", SVC(max_iter=1000000))
            ]),
            "normalized": Pipeline([
                ("preprocessing", Normalizer()),
                ("estimator", SVC(max_iter=1000000))
            ]),
            "scaled": Pipeline([
                ("preprocessing", MaxAbsScaler()),
                ("estimator", SVC(max_iter=1000000))
            ]),
        }

        datasets = {
            "tf": X,
            "tf_idf": transform_tf_to_tf_idf(X)
        }
        Cs = [10.0 ** x for x in range(-2, 5)]
        gammas = [10.0 ** x for x in range(-8, 1)] + ["scale"]
        param_grid = {
            "svm-rbf":
                {'estimator__kernel': ['rbf'],
                 'estimator__gamma': gammas,
                 'estimator__C': [10.0 ** x for x in range(-1, 10)]},
            "svm-linear":
                {'estimator__kernel': ['linear'],
                 'estimator__C': Cs},
            "svm-poly-d2-c00":
                {'estimator__kernel': ['poly'],
                 'estimator__C': Cs,
                 'estimator__degree': [2]},
            "svm-poly-d2-c01":
                {'estimator__kernel': ['poly'],
                 'estimator__C': Cs,
                 'estimator__degree': [2],
                 'estimator__coef0': [1]},
            "svm-poly-d3-c00":
                {'estimator__kernel': ['poly'],
                 'estimator__C': Cs,
                 'estimator__degree': [3]},
            "svm-poly-d3-c01":
                {'estimator__kernel': ['poly'],
                 'estimator__C': Cs,
                 'estimator__degree': [3],
                 'estimator__coef0': [1]},
            "gnb-sk":
                {'estimator': [GaussianNB()]},
            "gnb-my":
                {'estimator': [GaussianNaiveBayes()]},
            "multinomialnb":
                {'estimator': [MultinomialNB()]}
        }


        results = {dataset:{ pipeline:{ group: {} for group in param_grid.keys()} for pipeline in pipelines.keys()} for dataset in datasets.keys()}
        param_search_cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
        eval_cv = StratifiedKFold(n_splits=10,shuffle=True, random_state=42)
        progress = np.zeros(3,dtype=np.uint16)
        for dataset_name, dataset in datasets.items():
            progress = progress*np.array([1,0,0], dtype=np.uint16) + np.array([1,0,0], dtype=np.uint16)
            print(f"Processing dataset {progress[0]} of {len(datasets)}")
            for pipeline_name, pipeline in pipelines.items():
                progress = progress*np.array([1,1,0], dtype=np.uint16) + np.array([0,1,0], dtype=np.uint16)
                print(f"Processing pipeline {progress[1]} of {len(pipelines)}")
                for group_name, group in param_grid.items():
                    progress = progress * np.array([1, 1, 1], dtype=np.uint16) + np.array([0, 0, 1], dtype=np.uint16)
                    print(f"Processing group {progress[2]} of {len(param_grid)}")
                    param_search = GridSearchCV(pipeline, group, n_jobs=-1, cv=param_search_cv, verbose=1)
                    cv_results = cross_validate(param_search, dataset, y, cv=eval_cv, return_estimator=True, verbose=1)

                    results[dataset_name][pipeline_name][group_name] = cv_results

        with open(pickled_train_set_results, 'wb') as file:
            pickle.dump(results, file)

    task = "experiment"
    if task == "table_1":
        table_1(results)
    elif task == "table_2":
        table_2(results)
    elif task == "table_3":
        table_3(results)
    elif task == "table_4":
        table_4(results)
    elif task == "table_5":
        table_5(results)
    elif task == "table_6":
        table_6(results)
    elif task == "table_7":
        table_7(results)
    elif task == "table_8":
        table_8(results)
    elif task == "figure_1":
        figure_1(results, X, y)
    elif task == "experiment":
        experiment(X,y)
    elif task == "experiment2":
        experiment2(X,y)
    elif task == "histograms":
        plot_feature_histogram(minmax_scale(X), y, feature=1)
    elif task == "plot_2d":
        plot_2_dim_of_data(minmax_scale(X),y)

def table_1(train_set_results):
    """Table of scores of RBF kernel SVM with varying C and Gamma"""
    print_table_of_2_params(train_set_results['tf']['plain']['svm-rbf'], 'estimator__C','estimator__gamma',"C vs gama")
def table_2(train_set_results):
    """Table showcases rising training times with higher C"""
    print_results(train_set_results,
                             list(train_set_results.keys()),
                             ['plain', 'normalized', 'scaled'],
                             ["svm-rbf", "svm-linear", "svm-poly-d2-c00"],
                                                           "mean_fit_time",
                             apply_filter=lambda x: x[0].get('estimator__coef0',0) == 0 and x[0].get('estimator__gamma',0.001) == 0.001,
                             title="Table of mean fit time"
                             )
def table_3(train_set_results):
    """Table showcases rising training times with higher C"""
    print_results(train_set_results,
                             list(train_set_results.keys()),
                             ['plain', 'normalized', 'scaled'],
                             ["svm-rbf", "svm-linear", "svm-poly-d2-c00"],
                                                           "mean_score_time",
                             apply_filter=lambda x: x[0].get('estimator__gamma',0.001) == 0.001,
                             title="Table of mean score time"
                             )
def table_4(train_set_results):
    """Table showcases rising training times with higher C"""
    print_results(train_set_results,
                             list(train_set_results.keys()),
                             ['plain', 'normalized', 'scaled'],
                             ["svm-rbf", "svm-linear", "svm-poly-d2-c00"],
                                                           "mean_test_score",
                             as_percentage=True,
                             apply_filter=lambda x: x[0].get('estimator__gamma',0.001) == 0.001,
                             title="Table of mean score"
                             )
def table_5(results):
    "Show coef0 0 vs 1"
    print_results(results,
                             list(results.keys()),
                             ['plain', 'normalized', 'scaled'],
                             ["svm-linear", "svm-poly-d2-c00", "svm-poly-d2-c01"],
                                                        "mean_test_score",
                             as_percentage=True,
                             apply_filter=lambda x: x[0].get('estimator__gamma',
                                                                                        0.001) == 0.001,
                             title="Table of mean score"
                             )

def table_6(results):
    "Show coef0 0 vs 1"
    print_results(results,
                             list(results.keys()),
                             ['plain', 'normalized', 'scaled'],
                             ["svm-linear", "svm-poly-d2-c00", "svm-poly-d2-c01"],
                                                        "mean_fit_time",
                             title="Table of mean fitting time"
                             )

def table_7(results):
    "Show coef0 0 vs 1"
    print_results(results,
                  list(results.keys()),
                  ['plain', 'normalized', 'scaled'],
                  results['tf']['plain'].keys(),
                  "test_score",
                  title="Table of mean fitting time",
                  scope="crossvalidation",
                  as_percentage = True,
                  )

def table_8(results):
    "Show coef0 0 vs 1"
    print_results(results,
                  list(results.keys()),
                  ['plain', 'normalized', 'scaled'],
                  results['tf']['plain'].keys(),
                  "test_score",
                  title="Table of mean fitting time",
                  scope="crossvalidation",
                  as_percentage = True,
                  calculate_statistic= scipy.stats.sem
                  )


def figure_1(train_set_results, X_test,y_test):
    best_SVM = max([(group_results['best_score'],group_results['best'])for group_results in train_set_results['tf'].values()],
                   key=lambda x: x[0])
    plot_learning_curves_macro(SVC(**best_SVM[1]),GaussianNaiveBayes(),"SVM", "GNB", "SVM vs GNB training curves",X_test,y_test, e1_is_slow=True)

def experiment(X,y):
    X_bin = binarize(X)
    eval_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print("GaussianNB")
    res = cross_val_score(GaussianNB(),X,y,cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("GaussianNB + idf")
    tfpipe = Pipeline([("idf", TFtoTF_IDF_trnasformer()), ("nb", GaussianNB())])
    res = cross_val_score(tfpipe, X, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("Multinominal +idf")
    tfpipe = Pipeline([("idf", TFtoTF_IDF_trnasformer()), ("nb", MultinomialNB())])
    res = cross_val_score(tfpipe, X, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("SVM + idf")
    tfpipe = Pipeline([("idf", TFtoTF_IDF_trnasformer()), ("svm", SVC(C=10000))])
    res = cross_val_score(tfpipe, X, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("MultinominaNB")
    res = cross_val_score(MultinomialNB(), X, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("GaussianNB binarized")
    res = cross_val_score(GaussianNB(),X_bin,y,cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("MultinominalNB binarized")
    res = cross_val_score(MultinomialNB(), X_bin, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("BernoulliNB binarized")
    res = cross_val_score(BernoulliNB(), X_bin, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))





def plot_feature_histogram(X,y, feature=0):
    X = normalize(X)
    filter_mask= y==0
    X_class0 = X[filter_mask,feature]
    filter_mask= y==1
    X_class1 = X[filter_mask, feature]

    filter_0=True
    if filter_0:
        filter_mask = [i!=0 for i in X_class0]
        X_class0 = X_class0[filter_mask]

        filter_mask = [i != 0 for i in X_class1]
        X_class1 = X_class1[filter_mask]


    #plt.ylim(0,200)
    plt.hist((X_class0, X_class1), bins=[-0.01,0.00001]+[0.02*i for i in range(1,40)],density=True)

    x = np.linspace(-0.01, max(X_class1.max(), X_class0.max()), 200)
    plt.plot(x, scipy.stats.norm.pdf(x, X_class0.mean(), X_class0.std()))
    plt.plot(x, scipy.stats.norm.pdf(x, X_class1.mean(), X_class1.std()))
    plt.show()

def plot_2_dim_of_data(X,y):
    #X=normalize(X)
    X_reduced = X[:,[2,5,10]]
    X_reduced = normalize(X_reduced)
    fig = plt.figure()
    if True:
        ax = fig.add_subplot(projection='3d')
        #ax.set_aspect('equal')
        ax.scatter(X_reduced[:,0],X_reduced[:,1],X_reduced[:,2],c=y)
        plt.show()

def experiment2(X,y):
    filter_mask = y == 0
    X_class0 = X[filter_mask]
    filter_mask = y == 1
    X_class1 = X[filter_mask]

    X_class0 = binarize(X_class0)
    X_class1 = binarize(X_class1)

    X_class0 = X_class0.sum(0)
    X_class1 = X_class1.sum(0)

    print(X_class0)
    print()
    print(X_class1)

main()