# SupervisedLearningGT
Assignment 1: SupervisedLearning assignment for GT masters

Objective:
This report aims to provide an analysis of the supervised learning algorithms based on empirical data obtained through experiments on two datasets described below. The algorithms surveyed include:
a.	Decision Trees
b.	K Nearest Neighbors
c.	Neural Networks
d.	Boosting and
e.	Support Vector Machines.

The analysis pattern includes identifying the learning curve of an algorithm followed by identifying the validation curves (Model complexity) for two of the applicable hyperparameters. Based on the information seen in the model complexity I then seek to seed the algorithm with the optimal hyperparameters such as to generate an optimal model using the algorithm. The performance of the optimal model is then evaluated by testing against a test dataset and the corresponding confidence matrix/scatter plot is shown.

This repository contains the code that was used to derive the analysis described above.

Code Structure:
The code in this repository consists of several files, one each for a combination of the dataset and the algorithim that it works on.

root dir
	|
	|-ANNLearningAbalone.py
	|-ANNLearningLetter.py
	|-BoostingLearningAbalone.py
	|-BoostingLearningLetter.py
	|-DecisionTreeLearningAbalone.py
	|-DecisionTreeLearningLetter.py
	|-KNNLearningAbalone.py
	|-KNNLearningLetter.py
	|-SVMLearningAbalone.py
	|-SVMLearningLetter.py
	|-timing.py
	|-run.py

How to run this code:
Ensure you have python and pip installed.
Install dependencies using the "requirements.txt" file included in this repo.

The entry point for this code is through run.py
Here are the args that may be used when you run it.
--dt 				-> use to include decision tree for analysis
--boost				-> use to include boosted decision tree for analysis
--knn				-> use to include k nearest neighbour for analysis
--ann				-> use to include multi layer perceptron for analysis
--svm				-> use to include support vector machine for analysis
--generateModel		-> In addition to one/many of the parameters above, use this to generate the final learning, timing and accuracy plots
					   You will also need to modify the parameters to pass to the model in the "generateFinalModel"	method of the appropriate file.
					   
--generateGraph		-> In addition to one/many of the parameters above, use this to generate the validation curve plots on the base estimator

--search			-> In addition to one/many of the parameters above, use this to do grid search on the base model
					   You will also need to modify the parameters to pass to the model in the "doGridSearch"	method of the appropriate file.
					   
	For example: run --dt --generateGraphs
