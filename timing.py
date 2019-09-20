import numpy as np
from sklearn.model_selection import train_test_split
from time import clock
import pandas as pd
from matplotlib import pyplot as plt

def plot_model_timing(title, data_sizes, fit_scores, predict_scores, ylim=None):
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Data Size (% of total)")
    plt.ylabel("Time (s)")
    fit_scores_mean = np.mean(fit_scores, axis=1)
    fit_scores_std = np.std(fit_scores, axis=1)
    predict_scores_mean = np.mean(predict_scores, axis=1)
    predict_scores_std = np.std(predict_scores, axis=1)
    plt.grid()
    plt.tight_layout()

    plt.fill_between(data_sizes, fit_scores_mean - fit_scores_std,
                     fit_scores_mean + fit_scores_std, alpha=0.2)
    plt.fill_between(data_sizes, predict_scores_mean - predict_scores_std,
                     predict_scores_mean + predict_scores_std, alpha=0.2)
    plt.plot(data_sizes, fit_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Fit time")
    plt.plot(data_sizes, predict_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Predict time")

    plt.legend(loc="best")
    return plt

def getTimingData(X, y, classifier, algoname, datasetName, prefix=''):
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tests = 5
    out = dict()
    out['train'] = np.zeros(shape=(len(sizes), tests))
    out['test'] = np.zeros(shape=(len(sizes), tests))
    for i, val in enumerate(sizes):
        for j in range(tests):
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - val, random_state=np.random.seed(0))
            st = clock()
            classifier.fit(x_train, y_train)
            out['train'][i, j] = (clock() - st)
            st = clock()
            classifier.predict(x_test)
            out['test'][i, j] = (clock() - st)

    train_df = pd.DataFrame(out['train'], index=sizes)
    test_df = pd.DataFrame(out['test'], index=sizes)
    plt = plot_model_timing('{} - {}'.format(datasetName, algoname),
                            np.array(sizes) * 100, train_df, test_df)
    filename = '{}/images/{}/{}/{}_{}_timing_{}.png'.format('.', datasetName, algoname,
                                                                        datasetName, algoname, prefix)
    plt.savefig(filename, format='png', dpi=150)
    plt.close()