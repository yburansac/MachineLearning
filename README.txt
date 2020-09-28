The main.py file contain almost all the table data generating functionality, the task is selected by manualy
editing the task variable in the main function. It can reuse the precomputed results contained in gridsearch.7z
if extracted. Otherwise it will recompute the tests.

param_effects_plotter.py can produce the 2d decision boundary plots for the toy datasets.

naive_bayes.py implements the Gaussian naive Bayes classifier.

Other files are just helper functions and parts of the SPAMBASE dataset.

Installing packages listed in requirements.txt via pip should contain all necessary and unnecessary dependencies
(THX matplotlib)