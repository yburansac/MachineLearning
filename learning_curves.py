import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    # if axes is None:
    #     _, axes = plt.subplots(1, 1, figsize=(20, 5))
    _, axes = plt.subplots()
    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.set_xlim(left=0)
    axes.legend(loc="best")

    return plt

def plot_comarison_of_learing_curves(estimator1, estimator2, e1_name, e2_name, title, X, y, cv1, cv2,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    _, axes = plt.subplots()
    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes1, _, test_scores1 = \
        learning_curve(estimator1, X, y, cv=cv1, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       )
    train_sizes2, _, test_scores2 = \
        learning_curve(estimator2, X, y, cv=cv2, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       )
    test_scores1_mean = np.mean(test_scores1, axis=1)
    test_scores1_std = np.std(test_scores1, axis=1)
    test_scores2_mean = np.mean(test_scores2, axis=1)
    test_scores2_std = np.std(test_scores2, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes1, test_scores1_mean - test_scores1_std,
                      test_scores1_mean + test_scores1_std, alpha=0.2,
                      color="r")
    axes.fill_between(train_sizes2, test_scores2_mean - test_scores2_std,
                      test_scores2_mean + test_scores2_std, alpha=0.2,
                      color="g")
    axes.plot(train_sizes1, test_scores1_mean, 'o-', color="r",
              label=e1_name)
    axes.plot(train_sizes2, test_scores2_mean, 'o-', color="g",
              label=e2_name)
    axes.set_xlim(left=0)
    axes.legend(loc="best")

    return plt
