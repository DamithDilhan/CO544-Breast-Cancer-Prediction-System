Data set :- https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

Relavent paper :- 
- article 1

    https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3877-1
- article 2

    https://link.springer.com/article/10.1007/s12020-016-0893-x


From article 1

`
Methods
For each of the 166 participants several clinical features were observed or measured, including age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin and MCP-1. Machine learning algorithms (logistic regression, random forests, support vector machines) were implemented taking in as predictors different numbers of variables. The resulting models were assessed with a Monte Carlo Cross-Validation approach to determine 95% confidence intervals for the sensitivity, specificity and AUC of the models.
`

packages:
- gradio

references:
    - https://machinelearningmastery.com/feature-selection-machine-learning-python/
    - https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
    - https://practicaldatascience.co.uk/machine-learning/how-to-create-a-linear-regression-model-using-scikit-learn
    - https://bmccancer.biomedcentral.com/track/pdf/10.1186/s12885-017-3877-1.pdf
	- https://www.displayr.com/what-is-a-roc-curve-how-to-interpret-it/
    - https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
    - https://medium.com/analytics-vidhya/three-steps-in-case-of-imbalanced-data-and-close-look-at-the-splitter-classes-8b73628a25e6
    - https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2

Overfitting
- since we have small dataset and we try to learn from it.
- higher number of features
- feature selection using F test anf mutual test in k best 
- https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2

- solution :
    - feature selection

    - grid search for suitable model 
        used models : ....
    - choose kneighbour 
    - use Stratified cross validation due to imbalnce data , equal precentage to every split
     https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
    - k = 5 , train 2:45 , 1:36 , test: 2:19 ,1: 16 

    final model - train : 81 test :35
    roc curve

    116 , healthy 62 , patient 54 
    


