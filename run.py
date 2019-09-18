import argparse

from DecisionTreeLearningBC import decisionTreeLearnerBC
from DecisionTreeLearningLetter import decisionTreeLearnerLetter
from DecisionTreeLearningAbalone import decisionTreeLearnerAbalone
from DecisionTreeLearningTraffic import decisionTreeLearnerTraffic

from KNNLearningBC import knnLearnerBC
from ANNLearningBC import annLearnerBC


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify learning algorithm')
    parser.add_argument('--dt', action='store_true', help='DT')
    parser.add_argument('--knn', action='store_true', help='KNN')
    parser.add_argument('--ann', action='store_true', help='ANN')
    parser.add_argument('--boosting', action='store_true', help='Boosting')
    parser.add_argument('--svm', action='store_true', help='SVM')
    parser.add_argument('--bc', action='store_true', help='Run on BC dataset')
    parser.add_argument('--ab', action='store_true', help='Run on Abalone dataset')
    parser.add_argument('--allAlgo', action='store_true', help='Run all algo on all dataset')
    parser.add_argument('--allDataset', action='store_true', help='Run all algo on all dataset')

    args = parser.parse_args()
    alllearners = set()
    if args.allDataset and args.allAlgo:
        alllearners.add(decisionTreeLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv'))
        alllearners.add(decisionTreeLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv'))
        alllearners.add(decisionTreeLearnerTraffic('./Datasets/Interstate Traffic/Metro_Interstate_Traffic_Volume.csv'))
        alllearners.add(decisionTreeLearnerAbalone('./Datasets/Abalone/abalone.csv'))


    # elif args.allDataset:
    #     if args.bc:
    #         learner = decisionTreeLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')
    #     elif args.letter:
    #         learner = decisionTreeLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv')
    #     elif args.letter:
    #         learner = decisionTreeLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv')
    #
    #
    # if args.dt or args.all:
    #     if args.bc:
    #         learner = decisionTreeLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')
    #     elif args.ab:
    #         learner = decisionTreeLearnerLetter('./Datasets/LetterRecognition/letter-recognition.csv')
    #
    # if args.knn:
    #     learner = knnLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')
    #
    # if args.ann:
    #     learner = annLearnerBC('./Datasets/Breast Cancer Classification/breast-cancer-wisconsin.csv')

    for lrnr in alllearners:
        lrnr.loadData()
        lrnr.learn()