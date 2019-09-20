import argparse

from DecisionTreeLearningBC import decisionTreeLearnerBC
from DecisionTreeLearningLetter import decisionTreeLearnerLetter
from DecisionTreeLearningAbalone import decisionTreeLearnerAbalone
from KNNLearningBC import knnLearnerBC
from KNNLearningAbalone import knnLearnerAbalone
from KNNLearningLetter import knnLearnerLetter
from BoostingLearningBC import boostingLearnerBC
from BoostingLearningAbalone import boostingLearnerAbalone
from BoostingLearningLetter import boostingLearnerLetter
from ANNLearningBC import annLearnerBC
from ANNLearningAbalone import annLearnerAbalone
from ANNLearningLetter import annLearnerLetter
from SVMLearningBC import SVMLearnerBC
from SVMLearningAbalone import SVMLearnerAbalone
from SVMLearningLetter import SVMLearnerLetter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify learning algorithm')
    # parser.add_argument('--dt', action='store_true', help='DT')
    # parser.add_argument('--knn', action='store_true', help='KNN')
    # parser.add_argument('--ann', action='store_true', help='ANN')
    # parser.add_argument('--boosting', action='store_true', help='Boosting')
    # parser.add_argument('--svm', action='store_true', help='SVM')
    parser.add_argument('--bc', action='store_true', help='Run on BC dataset')
    parser.add_argument('--ab', action='store_true', help='Run on abalone dataset')
    parser.add_argument('--ltr', action='store_true', help='Run on letter dataset')
    parser.add_argument('--generateGraph', action='store_true', help='Learn on all dataset')
    parser.add_argument('--generateModel', action='store_true', help='Final model on all dataset')
    parser.add_argument('--search', action='store_true', help='Grid search')

    args = parser.parse_args()
    alllearners = set()
    if args.bc:
        alllearners.add(annLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv'))
        alllearners.add(boostingLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv'))
        alllearners.add(knnLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv'))
        alllearners.add(decisionTreeLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv'))
        alllearners.add(SVMLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv'))
    if args.ltr:
        alllearners.add(annLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        alllearners.add(boostingLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        alllearners.add(knnLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        alllearners.add(decisionTreeLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        alllearners.add(SVMLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
    if args.ab:
        alllearners.add(annLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        alllearners.add(boostingLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        alllearners.add(knnLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        alllearners.add(decisionTreeLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        alllearners.add(SVMLearnerAbalone('./Datasets/Abalone/abalone.csv'))

    for lrnr in alllearners:
        lrnr.loadData()
        if args.generateGraph:
            lrnr.learn()
        if args.search:
            lrnr.doGridSearch()
        if args.generateModel:
            lrnr.generateFinalModel()
