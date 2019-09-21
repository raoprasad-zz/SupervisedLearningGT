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
    # algo
    parser.add_argument('--dt', action='store_true', help='DT')
    parser.add_argument('--knn', action='store_true', help='KNN')
    parser.add_argument('--ann', action='store_true', help='ANN')
    parser.add_argument('--boost', action='store_true', help='Boosting')
    parser.add_argument('--svm', action='store_true', help='SVM')
    # datasets
    parser.add_argument('--ab', action='store_true', help='Run on abalone dataset')
    parser.add_argument('--ltr', action='store_true', help='Run on letter dataset')
    # graph, model or search
    parser.add_argument('--generateGraph', action='store_true', help='Learn on all dataset')
    parser.add_argument('--generateModel', action='store_true', help='Final model on all dataset')
    parser.add_argument('--search', action='store_true', help='Grid search')

    args = parser.parse_args()
    alllearners = set()

    if args.ltr:
        if args.ann:
            alllearners.add(annLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        if args.boost:
            alllearners.add(boostingLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        if args.knn:
            alllearners.add(knnLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        if args.dt:
            alllearners.add(decisionTreeLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        if args.svm:
            alllearners.add(SVMLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
    if args.ab:
        if args.dt:
            alllearners.add(decisionTreeLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        if args.boost:
            alllearners.add(boostingLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        if args.ann:
            alllearners.add(annLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        if args.knn:
            alllearners.add(knnLearnerAbalone('./Datasets/Abalone/abalone.csv'))
        if args.svm:
            alllearners.add(SVMLearnerAbalone('./Datasets/Abalone/abalone.csv'))

    for lrnr in alllearners:
        lrnr.loadData()
        if args.generateGraph:
            lrnr.learn()
        if args.search:
            lrnr.doGridSearch()
        if args.generateModel:
            lrnr.generateFinalModel()
