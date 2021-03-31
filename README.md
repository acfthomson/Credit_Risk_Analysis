# Credit_Risk_Analysis

## Overview
Data analysts were asked to examine credit card data from peer-to-peer lending services company LendingClub in order to determin credit risk.  Supervised machine learning was employed in order to determine which model would perform the best against an unbalanced dataset.  Data analysts trained and evaluated several models to predict credit risk. 


## Dependencies
- [LoanStats_2019Q1.csv](https://github.com/acfthomson/Credit_Risk_Analysis/tree/main/Resources)
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

An average recall score of 0.64 means that 64% of class predictions were correct and 36% were incorrect.

Comparing the performance of the naive random oversampling and SMOTE oversampling models, they appeared to perform about the same.




