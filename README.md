<<<<<<< HEAD
# CO544 Group 09
#  Breast Cancer Prediction with Machine Learning 

## Project code
***
### Requirments:

-   python=3.9.5

-   matplotlib=3.4.2

-   numpy=1.20.3

-   pandas=1.3.2

-   scikit-learn=0.24.2

-   scikit-plot=0.3.7

***
### Folders

-   feature_selections

    o   fclassif.py - use score function as F test to select features

    o   mutual_classif.py - use score function as mutual test to select features

-   suitable_mls

    o   gridSearch.py  - grid search with all features

    o   gridSearch_selected_features.py - grid search with selected features 
    
    o   multi_classifier.py - Evalute classifiers with grid search results

- knn

    o   cv_run.py - KNN classifier with cross-validation

    o final_run.py - Final KNN classifier  

***
Before run any code dataset path need to be configure
in every file


```
DATA_PATH = os.path.join(path, "../data/dataR2.csv")
``` 







