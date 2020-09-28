import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC

# this utility allows some experimentation with the kernels in 2D, the predefined parameter sets at he end this file
# should produce the figures in the report, but the figure numbering doesn't match since the report has changed since
# writing this script


def is_matplotlib_just_insane_or_is_it_me(target_aspect_ratio: float, x_min: int, x_max: int, y_min: int, y_max: int):
    """
    Helps one actually control the shape of a generated plots.
    This function takes in coordinates of a bounding rectangle (axis aligned) and returns it scaled in
    one direction so it meets the target ratio.
    """
    x_size, y_size = x_max - x_min, y_max - y_min
    current_aspect_ratio=x_size/y_size
    if current_aspect_ratio > target_aspect_ratio: # rectangle is wider than intended scale y
        correction_factor = current_aspect_ratio/target_aspect_ratio
        y_cor = ((y_size*correction_factor) - y_size)/2
        y_min, y_max = y_min - y_cor, y_max + y_cor
    elif current_aspect_ratio < target_aspect_ratio: # rectangle is taller than intended scale x
        correction_factor = target_aspect_ratio / current_aspect_ratio
        x_cor = ((x_size * correction_factor) - x_size) / 2
        x_min, x_max = x_min - x_cor, x_max + x_cor
    return x_min, x_max, y_min, y_max


def to_plot_or_not_to_plot( # that is the question!
        studied_param="degree",
        samples=100,
        add_column_of=None,
        window_title="Some plot",
        coef0=0,
        use_datasets=None,
        class_separability=1.1,
        plot_margins=False,
        kernel_override="linear",
):
    """
    Plots a series of plots which aim at visualisation of the effect on a decision boundary a change of a parameter of
    a C-SVM has, using several toy 2D datasets. The decision boundaries are approximated by evaluation of the decision
    function on each point of a fine grid.
    """

    h = .01  # step size in the mesh

    params={
        "kernel": ['linear', 'poly', 'rbf'],
        "C" : [1, 5, 10],
        "degree": [2, 3],
        "gamma": [0.1, 2, 25]
    }

    names = [str(i) for i in params[studied_param]]

    classifiers = []
    for i in params[studied_param]:
        if studied_param == 'degree':
            classifiers.append(SVC(kernel='poly', C=1, coef0=coef0, degree=i))
        if studied_param == 'kernel':
            classifiers.append(SVC(kernel=i, C=1, coef0=coef0, degree=2))
        if studied_param == 'C':
            classifiers.append(SVC(kernel=kernel_override, C=i, coef0=coef0))
        if studied_param == 'gamma':
            classifiers.append(SVC(kernel='rbf', C=1, gamma=i, coef0=coef0))


    X, y = make_classification(n_samples=samples, n_features=2, n_redundant=0, n_informative=2,
                               random_state=21, n_clusters_per_class=1, class_sep=class_separability)
    rng = np.random.RandomState(2)
    X += 0 * rng.uniform(size=X.shape)
    X -= 0 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    moons = make_moons(n_samples=samples, noise=0.2, random_state=0)
    circles = make_circles(n_samples=samples, noise=0.15, factor=0.5, random_state=1)

    all_datasets = {"moons": moons,
                "circles": circles,
                "linearly_separable": linearly_separable}
    if use_datasets == None:
        use_datasets = all_datasets.keys()
    datasets = [all_datasets[dataset_name] for dataset_name in use_datasets]

    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels

    nrows, ncols = len(datasets), len(classifiers)+1
    dx, dy = 4, 3
    figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))

    fig, subplot_list = plt.subplots(len(datasets),len(classifiers)+1,
                                     sharex='row', sharey='row', figsize=figsize, num=window_title)

    i = 0
    top_row = True
    # iterate over datasets
    for X, y in datasets:
        # preprocess dataset, split into training and test part
        X = StandardScaler().fit_transform(X)
        if add_column_of is not None:
            X = np.c_[X, np.full(len(X),add_column_of)]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        view_area_margin = 2
        x_size, y_size = X[:, 0].max() - X[:, 0].min(), X[:, 1].max() - X[:, 1].min()
        x_min, x_max = X[:, 0].min() - 0.1*x_size, X[:, 0].max() + 0.1*x_size
        y_min, y_max = X[:, 1].min() - 0.1*y_size, X[:, 1].max() + 0.1*y_size

        x_min, x_max, y_min, y_max = is_matplotlib_just_insane_or_is_it_me(4.0/3, x_min, x_max, y_min, y_max)

        xx, yy = np.meshgrid(np.arange(x_min - view_area_margin, x_max + view_area_margin, h),
                             np.arange(y_min - view_area_margin, y_max + view_area_margin, h))

        # just plot the dataset first
        cm = ListedColormap(['#DE6B6B', '#2793B6'])
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm_alt = ListedColormap(['#f58251', '#51e4f5'])
        ax = subplot_list[i//(len(classifiers)+1),i%(len(classifiers)+1)] if len(datasets)>1 else subplot_list[i]
        i = i+1
        if top_row:
            ax.set_title(studied_param.upper() + ":")
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_alt, edgecolors='black', alpha=0.6)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='black')
        ax.set_aspect('equal', share=True)
        ax.tick_params(axis='both', which='major', pad=2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = subplot_list[i//(len(classifiers)+1),i%(len(classifiers)+1)] if len(datasets)>1 else subplot_list[i]
            i = i+1
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            ax.tick_params(axis='both', which='major', pad=2)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                try:
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                except ValueError:
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), np.full(len(xx.ravel()),add_column_of)])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.pcolormesh(xx, yy, Z > 0, cmap=cm, shading="auto")
            if plot_margins:
                ax.contour(xx, yy, Z, colors=['k', 'k', 'k'],
                           linestyles=['--', '-', '--'], levels=[-1.0, 0, 1.0])
            else:
                ax.contour(xx, yy, Z, colors='black', levels=[0])


            # Plot support vectors
            ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], facecolors='none', zorder=2, edgecolors='gold',linewidths=2)
            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='black', s=25, zorder=3)

            if top_row:
                ax.set_title(name.upper())
            ax.text( 0.95,0.08,('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right', transform=ax.transAxes)

        top_row=False
    plt.tight_layout(pad=0)
    plt.show()

# these combinations of parameters correspond to figures in the report
scenarios = {
    "Figure_1":{
        "window_title": "Figure_1",
        "studied_param": "kernel",
    },
    "Figure_2a":{
        "window_title": "Figure_2a",
        "studied_param": "degree",
    },
    "Figure_2b":{
        "window_title": "Figure_2b",
        "studied_param": "degree",
        "coef0": 1
    },
    "Figure_2c":{
        "window_title": "Figure_2c",
        "studied_param": "degree",
        "add_column_of": 1,
    },
    "Figure_3a":{
        "window_title": "Figure_3a",
        "studied_param": "C",
        "samples": 32,
        "use_datasets": ["linearly_separable"],
        "class_separability": 0.67,
        "plot_margins":True
    },
    "Figure_3b":{
        "window_title": "Figure_3b",
        "studied_param": "C",
        "kernel_override": "poly",
        "samples": 67,
        "use_datasets": ["linearly_separable"],
        "class_separability": 0.949,
        "coef0":1,
        "plot_margins":True
    },
    "Figure_3c":{
        "window_title": "Figure_3c",
        "studied_param": "C",
        "samples": 80,
        "use_datasets": ["moons"],
        "kernel_override": "rbf",
        "plot_margins":True
    },
    "Figure_3d":{
        "window_title": "Figure_3d",
        "studied_param": "gamma",
        "samples": 80,
        "use_datasets": ["moons"],
        "plot_margins": True
    }
}

to_plot_or_not_to_plot(**scenarios["Figure_3d"])
