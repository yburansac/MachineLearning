#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tabulate
import pickle
import scipy.stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit, StratifiedKFold

from sklearn.preprocessing import normalize, minmax_scale, Normalizer, MaxAbsScaler, binarize
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from naive_bayes import GaussianNaiveBayes

import statistics

# to use this script  select task by assigning value to task variable in the main function and then run


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


def print_table_of_2_params(results, param_x, param_y, title="Some table"):
    """
    Prints table from the results dictionary, currently used only to print the tables for the heatmaps in the
    report.
    """
    results = results['estimator'][0].cv_results_
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
        table[param_list_x.index(C)][param_list_y.index(gamma)] = 100*results["mean_test_score"][i]
    print_table_title(title)
    print(tabulate.tabulate(table, headers=param_list_y, showindex=param_list_x))


def print_results(results, dataset_names, pipeline_names, group_names, metric,
                  scope='gridsearch',  apply_filter=None, title="Some table", as_percentage=False,
                  calculate_statistic=statistics.mean, floatfmt=".2f"):
    """
    Prints a table dataset_names x pipeline names columns and group_names rows of choossen metric from the results
    dictionary at the summary (outer cv) or gridsearch (inner cv) scope
    :param results: the results as created in main
    :param dataset_names: [tf, idf]
    :param pipeline_names: [plain, normalized, scaled]
    :param group_names: [svm-rbf, svm-linear,  GNB-sk, ...]
    :param metric: [test_score, mean_fit_time ...]
    :param scope: [gridsearch, summary]
    :param apply_filter: a function on the parameter dicts -> Bool
    :param title: a label for the table
    :param as_percentage: multiplies the metric by 100
    :param calculate_statistic: a function to call at summary level where metrics are a list of values for each fold
    :param floatfmt: formating of floating point numbers
    :return:
    """
    index = []
    for group_name in group_names:
        if scope == 'gridsearch':
            params = results[dataset_names[0]][pipeline_names[0]][group_name]['estimator'][0].cv_results_['params']
        else:
            params = [group_name]
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
    """Prints table title centered within terminal width with lines under and over it"""
    print(terminal_width * '-')
    centering_offset = max(0, (terminal_width - len(title))//2)
    print(centering_offset * " " + title)
    print(terminal_width * '-')


def transform_tf_to_tf_idf(X):
    """Utility to convert TF to TF-IDF representation"""
    return TFtoTF_IDF_trnasformer().fit_transform(X)


class TFtoTF_IDF_trnasformer(TransformerMixin):
    """Transforms TF to TF-IDF representation with scikit compatible API"""
    def __init__(self):
        self.idf = None

    def fit(self, X, y=None):
        df = np.count_nonzero(X, 0)
        smoothing = 1
        n = X.shape[0] + smoothing
        df += smoothing
        self.idf = -np.log(df/n)
        return self

    def transform(self, X):
        return self.idf * X


def plot_learning_curves_macro(datset, dataset_labels, estimator1, estimator2=None, e1_name="e1", e2_name="e2",
                               title="Learning curve", e1_is_slow=False, e2_is_slow=False):
    """
    A convenience function for plotting learning curves
    :param estimator1: a classifier to be tested
    :param estimator2: a classifier to be tested [optional]
    :param e1_name: estimator displayed name
    :param e2_name: estimator displayed name
    :param title: Title for the figure
    :param datset: dataset
    :param dataset_labels: dataset labels
    :param e1_is_slow: if true does only 10 splits instead of 100 for this estimator
    :param e2_is_slow: if true does only 10 splits instead of 100 for this estimator
    :return:
    """
    from learning_curves import plot_learning_curve, plot_comarison_of_learing_curves
    cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) if e1_is_slow else ShuffleSplit(n_splits=100,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    cv2 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) if e2_is_slow else ShuffleSplit(n_splits=100,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    train_sizes = np.geomspace(0.01, 1.0, 15)
    if estimator2:
        plot_comarison_of_learing_curves(estimator1, estimator2, e1_name, e2_name, title, datset, dataset_labels,
                                         cv1=cv1, cv2=cv2, n_jobs=-1, train_sizes=train_sizes)
    else:
        plot_learning_curve(estimator1, title, datset, dataset_labels, cv=cv1, n_jobs=-1, train_sizes=train_sizes)
    plt.show()

def confusion_matrix(classifer, X_test,y_test, class_names, title):
    """Convinience helper function to plot confusion matrix"""
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=25)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    X_test= normalize(X_test)
    X_train, X_test, y_train, y_test = train_test_split(X_test,y_test, test_size=0.3, shuffle=True, random_state=42)
    classifer.fit(X_train,y_train)
    plot = plot_confusion_matrix(classifer,X_test,y_true=y_test,display_labels=class_names, normalize='true',cmap=plt.cm.Blues)
    plot.ax_.set_title(title)

    plt.show()

def main():
    """
    Loads the dataset and either computes or loads precomputed results and using them (or not) completes task
    designated by the task variable
    :return: Happiness
    """
    from pathlib import Path
    class_names = ["ham", "spam"]

    X, y = load_set()
    # X, _, y, _ = train_test_split(X,y,train_size=0.12) <- use to make dataset smaller for testing

    recompute = False

    pickled_train_set_results = "gridsearch.pickle"

    # load the precomputed results file if exists
    if Path(pickled_train_set_results).is_file() and not recompute:
        with open(pickled_train_set_results, 'rb') as file:
            results = pickle.load(file)
    else:
        # compute the test results (may take a long time)
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

        results = {dataset: {pipeline: {group: {} for group in param_grid.keys()} for pipeline in pipelines.keys()}
                   for dataset in datasets.keys()}
        param_search_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        eval_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        progress = np.zeros(3, dtype=np.uint16)
        for dataset_name, dataset in datasets.items():
            progress = progress*np.array([1, 0, 0], dtype=np.uint16) + np.array([1, 0, 0], dtype=np.uint16)
            print(f"Processing dataset {progress[0]} of {len(datasets)}")
            for pipeline_name, pipeline in pipelines.items():
                progress = progress*np.array([1, 1, 0], dtype=np.uint16) + np.array([0, 1, 0], dtype=np.uint16)
                print(f"Processing pipeline {progress[1]} of {len(pipelines)}")
                for group_name, group in param_grid.items():
                    progress = progress * np.array([1, 1, 1], dtype=np.uint16) + np.array([0, 0, 1], dtype=np.uint16)
                    print(f"Processing group {progress[2]} of {len(param_grid)}")
                    param_search = GridSearchCV(pipeline, group, n_jobs=-1, cv=param_search_cv, verbose=1)
                    cv_results = cross_validate(param_search, dataset, y, cv=eval_cv, return_estimator=True, verbose=1)

                    results[dataset_name][pipeline_name][group_name] = cv_results
        # save the results for future reuse
        with open(pickled_train_set_results, 'wb') as file:
            pickle.dump(results, file)

    # select your task
    task = "table_7" # <-----here

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
    elif task == "figure_2":
        figure_2(results, X, y)
    elif task == "figure_3":
        figure_3(results, X, y, class_names)
    elif task == "figure_4":
        figure_4(X, y, class_names)
    elif task == "figure_5":
        figure_5(X, y, class_names)
    elif task == "experiment":
        experiment(X, y)
    elif task == "histograms":
        plot_feature_histogram(minmax_scale(X), y, feature=7)
    else:
        print(f"Unknown task; {task}")


def table_1(train_set_results):
    """Table of scores of RBF kernel SVM with varying C and Gamma"""
    print_table_of_2_params(train_set_results['tf']['plain']['svm-rbf'], 'estimator__C', 'estimator__gamma',
                            "C vs gama")


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
    """Table of score times with relation to C"""
    print_results(train_set_results,
                  list(train_set_results.keys()),
                  ['plain', 'normalized', 'scaled'],
                  ["svm-rbf", "svm-linear", "svm-poly-d2-c00"],
                  "mean_score_time",
                  apply_filter=lambda x: x[0].get('estimator__gamma',0.001) == 0.001,
                  title="Table of mean score time"
                  )


def table_4(train_set_results):
    """Table of scores with relation to  C"""
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
    """Show coef0 0 vs 1 polynomial kernel accuracies"""
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
    """Show coef0 0 vs 1 polynomial kernel fitting times"""
    print_results(results,
                  list(results.keys()),
                  ['plain', 'normalized', 'scaled'],
                  ["svm-linear", "svm-poly-d2-c00", "svm-poly-d2-c01"],
                  "mean_fit_time",
                  title="Table of mean fitting time"
                  )

def table_7(results):
    """
    Summary (outer cv) mean accuracies across all datasets with all preprocessing options and classifier
    configurations
    """
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
    """
    Summary (outer cv) mean accuracies' standard error of the mean across all datasets with all preprocessing options
    and classifier configurations
    """
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


def table_10(results):
    """
    Summary (outer cv) mean fit time across all datasets with all preprocessing options and classifier configurations
    """
    print_results(results,
                  list(results.keys()),
                  ['plain', 'normalized', 'scaled'],
                  results['tf']['plain'].keys(),
                  "mean_fit_time",
                  title="Table of mean fitting time",
                  )


def figure_1(train_set_results, X_test,y_test):
    """Learning curves best SVMvs GNB"""
    best_SVM_params = max([(statistics.mean(group_results['test_score']),group_results['estimator'][0].best_params_)for group_results in train_set_results['tf']['plain'].values()],
                          key=lambda x: x[0])
    best_SVM = Pipeline([("estimator", SVC(max_iter=1000000))])
    best_SVM.set_params(**best_SVM_params[1])
    plot_learning_curves_macro(best_SVM,GaussianNaiveBayes(),"SVM", "GNB", "SVM vs GNB learning curves",X_test,y_test)


def figure_2(train_set_results, X_test,y_test):
    """Learning curve best SVM vs MultinomialNB normalized dataset"""
    X_test = normalize(X_test)
    best_SVM_params = max([(statistics.mean(group_results['test_score']),group_results['estimator'][0].best_params_)for group_results in train_set_results['tf']['normalized'].values()],
                          key=lambda x: x[0])
    best_SVM = Pipeline([("estimator", SVC(max_iter=1000000))])
    best_SVM.set_params(**best_SVM_params[1])
    plot_learning_curves_macro(best_SVM,MultinomialNB(),"SVM", "MultinomialNB", "SVM vs MultinomialNB learning curves",X_test,y_test)


def figure_3(train_set_results, X_test,y_test, class_names):
    """Confusion matrix for best SVM"""
    best_SVM_params = max(
        [(statistics.mean(group_results['test_score']), group_results['estimator'][0].best_params_) for group_results in
         train_set_results['tf']['normalized'].values()],
        key=lambda x: x[0])
    best_SVM = Pipeline([("estimator", SVC(max_iter=1000000))])
    best_SVM.set_params(**best_SVM_params[1])
    confusion_matrix(best_SVM,X_test,y_test,class_names,"SVM")


def figure_4(X_test,y_test, class_names):
    """Confusion matrix for GNB"""
    confusion_matrix(GaussianNB(), X_test,y_test,class_names,"Gaussian NB")


def figure_5(X_test, y_test, class_names):
    """Confusion matrix for MultinomialNB"""
    confusion_matrix(MultinomialNB(), X_test, y_test, class_names, "Multinomial NB")


def experiment(X,y):
    """
    Test multiple NB classifiers over TF representation and its normalized and binarized transforms
    """
    X_bin = binarize(X)
    X_nor = normalize(X)
    eval_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print("GaussianNB")
    res = cross_val_score(GaussianNB(),X,y,cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("MultinominaNB")
    res = cross_val_score(MultinomialNB(), X, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("GaussianNB normalized")
    res = cross_val_score(GaussianNB(), X_nor, y, cv=eval_cv, n_jobs=-1)
    print(statistics.mean(res))
    print("MultinominaNB normalized")
    res = cross_val_score(MultinomialNB(), X_nor, y, cv=eval_cv, n_jobs=-1)
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
    """
    Plots a histogram of a feature with Gaussian PDF as learnt by the Gaussian NB overlayed
    """
    X = transform_tf_to_tf_idf(X)
    X = normalize(X)
    filter_mask= y==0
    X_class0 = X[filter_mask,feature]
    filter_mask= y==1
    X_class1 = X[filter_mask, feature]

    filter_0=False
    if filter_0:
        filter_mask = [i!=0 for i in X_class0]
        X_class0 = X_class0[filter_mask]

        filter_mask = [i != 0 for i in X_class1]
        X_class1 = X_class1[filter_mask]

    bins = [-0.02,0.000000000000001]+[0.02*i for i in range(1,51)]
    class0_weights = np.ones_like(X_class0) / len(X_class0)
    class1_weights = np.ones_like(X_class1) / len(X_class1)
    ax = plt.subplot()
    ax.set_title(label=f"Feature {feature}")
    ax.hist((X_class0,X_class1), weights=(class0_weights, class1_weights),bins=bins, label=("Class 0", "Class 1"))
    x = np.linspace(-0.01, max(X_class1.max(), X_class0.max()), 200)
    ax.plot(x, 0.02*scipy.stats.norm.pdf(x, X_class0.mean(), X_class0.std()+0.0000001), label="Class 0")
    ax.plot(x, 0.02*scipy.stats.norm.pdf(x, X_class1.mean(), X_class1.std()+0.0000001), label="Class 1")

    ax.legend()
    plt.show()

main()