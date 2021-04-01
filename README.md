# Credit_Risk_Analysis

## Overview
Data analysts were asked to examine credit card data from peer-to-peer lending services company LendingClub in order to determine credit risk. Supervised machine learning was employed to find out which model would perform the best against an unbalanced dataset. Data analysts trained and evaluated several models to predict credit risk. 


## Resources
- [LoanStats_2019Q1.csv](https://github.com/acfthomson/Credit_Risk_Analysis/tree/main/Resources)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/index.html)
- [scikit-learn documentation](https://scikit-learn.org/stable/supervised_learning.html)


## Dependencies
- Jupyter Notebook
- Python v3.x
  -  Dependencies
      -  Numpy
      -  Pandas
      -  Pathlib
      -  Collections
      -  SKLearn
      -  ImbalancedLearn


## Results
Credit card data from [LoanStats_2019Q1.csv](https://github.com/acfthomson/Credit_Risk_Analysis/tree/main/Resources) was cleaned prior to implementing machine learning techniques.  Null columns and rows were dropped, interest rates were converted to numerical values, and target (y-axis) columns were converted to low_risk and high_risk based on their values.

Once the data was cleaned, it was split into training and testing categories, which resulted in four sets of data:
- X_train
- X_test
- y_train
- y_test

A random_state of 1 was used across all models to ensure reproducible output.

The balance of low_risk and high_risk is unbalanced, but this was expected as credit risk is an inherently unbalanced classification problem, since good loans easily outnumber risky loans.

![dataset](https://user-images.githubusercontent.com/73897240/113198332-cb7ebc00-9233-11eb-9449-4769419781e8.PNG)

### Oversampling Algorithms
#### Naive Random Oversampling
In this model, instances of the high_risk class were oversampled, which is where data from the high_risk data set is randomly selected and added to the training set until the high_risk and low_risk classes were balanced.

Unbalanced                |  Balanced
:------------------------:|:-------------------------:
![unbalanced](https://user-images.githubusercontent.com/73897240/113197173-55c62080-9232-11eb-90bf-4c9c56d3e89e.PNG)|![balanced](https://user-images.githubusercontent.com/73897240/113197268-74c4b280-9232-11eb-98eb-59a0f91cb4eb.PNG)

Once the datasets were balanced, the model trained the data, which is where the algorithm analyzes the data and attempts to learn patterns in the data.

Naive random oversampling on this data gave the following scores:

Balanced Accuracy: 0.644

![class_report](https://user-images.githubusercontent.com/73897240/113199045-a8a0d780-9234-11eb-86ca-1ddc7752e66f.PNG)

A balanced accuracy score of 0.644 means that 35.6% of classes are incorrect and 64.4% are correct.

An average precision score of 0.99 means that this model quantified the number of positive class predictions that actually belong to the positive class 99% of the time.

An average recall score of 0.67 means that this model quantified the number of positive class predictions made out of all positive examples 67% of the time.


#### SMOTE Oversampling
In the Synthetic Minority Oversampling Technique (SMOTE) oversampling model, the minority class (high_risk) are duplicated prior to fitting the model.  This can balanced class distribution, but does not provide any additional information to the model.  SMOTE selects data points that are close in the feature space, drawing a line between the points in the feature space, and drawing a new sample at a point along that line.  Realistic data from high_risk are created , which are relatively close to existing data from high_risk.

Once the data were balanced and trained, SMOTE oversampling gave the following scores:

Balanced Accuracy: 0.648

![imb_class_report](https://user-images.githubusercontent.com/73897240/113201885-11d61a00-9238-11eb-8762-47c4877b858b.PNG)

The balanced accuracy score for this model means that 64.8% of classes are correct and 35.2% are incorrect.

An average precision score of 0.99 means that this model predicted positive class predictions 99% of the time.

An average recall score of 0.64 means that 64% of class predictions made out of all positive examples in the dataset were correct and 36% were incorrect.

Comparing the performance of the naive random oversampling and SMOTE oversampling models, they appeared to perform about the same.


### Undersampling Algorithm
#### ClusterCentroids
The ClusterCentroid algorithm provides an efficient way to represent the data cluster with a reduced number of samples.  A cluster is a group of data points grouped together because of certain similarities.  This algorithm does this by performing K-means clustering on the majority class, low_risk, and then creates new data points which are averages of the coordinates of the generated clusters.

Once the data were balanced and trained, ClusterCentroids undersampling gave the following scores:

Balanced Accuracy: 0.644

![class_report](https://user-images.githubusercontent.com/73897240/113206456-7a73c580-923d-11eb-8d64-76db77468089.PNG)

The balanced accuracy score for this model was 0.644, which means that 35.6% of classes are incorrect and 64.4% are correct.

An average precision score of 0.99 means the ClusterCentroid algorithm predicted positive class predictions 99% of the time on thie dataset.

An average recall score of 0.67 means that 67% of class predictions made out of all positive examples in the dataset were correct, whereas 33% were incorrect.


### Combination Sampling
#### SMOTEENN
The SMOTEENN algorithm is a combination of SMOTE and Edited Nearest Neighbor (ENN) algorithms.  In simple terms, SMOTEENN randomly oversamples the minority class (high_risk) and undersamples the majority class (low_risk) 

Once the data were balanced and trained, the SMOTEEN algorithm gave the following scores:

Balanced Accuracy: 0.644

![class_report](https://user-images.githubusercontent.com/73897240/113209206-c1af8580-9240-11eb-8ca7-89ff9fb0aff0.PNG)

SMOTEENN's balanced accuracy score was 0.644, which means 64.4% of class predictions were correct and 35.6% were incorrect.

An average precision score of 0.99 means the SMOTEENN algorithm predicted positive class predictions 99% of the time on this dataset.

An average recall score of 0.67 means that 67% of class predictions made out of all positive examples in the dataset were correct, whereas 33% were incorrect.


### Ensemble Learners
#### Balanced Random Forest Classifier
The Balanced Random Forest Classifier is an ensemble method where each tree in the ensemble is built from a sample drawn with replacement (bootstrap sample) from the training set. Instead of using all the features, a random subset of features is selected,  which further randomizes the tree.  As a result, the bias of the forest increases slightly, but since the less correlated trees are averaged, its variance decreases, which results in an overall better model.

Once the data were balanced and trained, the balanced random forest algorithm gave the following scores:

Balanced Accuracy: 0.788

![class_report](https://user-images.githubusercontent.com/73897240/113342656-be79cf80-92fc-11eb-9e79-9c3c7eb630f2.PNG)

This algorithm's balanced accuracy score is 0.788, which means nearly 79% of class predictions were correct and 21% were incorrect.

Balanced Random Forest's average precision score of 0.99 means that this algorithm predicted positive class predictions 99% of the time on this dataset.

An average recall score of 0.91 means that 91% of class predictions made out of all positive examples in this dataset were correct, where as 9% were incorrect.


#### Easy Ensemble AdaBoost Classifier
The Easy Ensemble AdaBoost Classifier combine multiple weak or low accuracy models to create a strong, accurate models.  This algorithm uses one-level decision trees as weak learners that are added to the ensemble sequentially.  This is an iterative process, so each subsequent model attempts to correct predictions made by the previous model in the sequence.

Once the data were balanced and trained, the Easy Ensemble AdaBoost Classifier algorithm gave the following scores:

Balanced Accuracy: 0.672

![class_report](https://user-images.githubusercontent.com/73897240/113345560-a0ae6980-9300-11eb-940d-0f8c62e24278.PNG)

Easy Ensemble AdaBoost Classifier's accuracy score of 0.925 means that its predictions were correct 92.5% of the time and 7.5% were incorrect.

This algorithm's precision score of 0.99 means that it predicted positive class predictions 99% of the time on this dataset.

The average recall score of 0.94 means that 94% of class predictions made out of all positive examples in this dataset were correct.  


## Summary
The oversampling, undersampling, and combination sampling algorithms' performance were relatively the same. Balanced Random Forest Classifier had a higher balanced accuracy score than the previous algorithms tested, but it was not good enough for predicting credit risk.

Out of the six supervised machine learning algorithms tested, Easy Ensemble AdaBoost CLassifier performed the best overall.  It had a balanced accuracy score, along with high precision and recall scores.  It also had a high specificity score, which means this algorithm correctly determined actual negatives 91% of the time, and a high F1 score.  This means the harmonic mean of precision and recall were 0.97 out of 1.0.

