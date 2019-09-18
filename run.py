import argparse

from DecisionTreeLearningBC import decisionTreeLearnerBC
from KNNLearningBC import knnLearnerBC
from ANNLearningBC import annLearnerBC


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify learning algorithm')
    parser.add_argument('--dt', action='store_true', help='DT')
    parser.add_argument('--knn', action='store_true', help='KNN')
    parser.add_argument('--ann', action='store_true', help='ANN')
    parser.add_argument('--boosting', action='store_true', help='Boosting')
    parser.add_argument('--svm', action='store_true', help='SVM')

    args = parser.parse_args()
    if args.dt:
        dtlearner = decisionTreeLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')
        dtlearner.loadBCData()
        dtlearner.learn()

    if args.knn:
        knnlearner = knnLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')
        knnlearner.loadBCData()
        knnlearner.learn()

    if args.ann:
        annlearner = annLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')
        annlearner.loadBCData()
        annlearner.learn()

    # if args.boosting:
    #
    #
    # if args.svm:
