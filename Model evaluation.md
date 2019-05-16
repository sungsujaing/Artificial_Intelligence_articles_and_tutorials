# Model evaluation and evalution metrics

During the model building process, understanding the model performance and setting an appropriate evaluation metric are critical as they can guide how to improve the model performance.

When debugging the learning algorithm, a few options to consider include:

* Obtain more training data (increase **_m_**)
* Add more features (increase **_n_**) through feature extraction
* Remove some features (decrease **_n_**) through feature selection
* Adjust hyperparameters (i.e. regularization parameters, type of kernels)

## In general: Overfitting (high variance) vs. Underfitting (high bias)

#### Rule of thumb: 

1. Model is to be fitted on the training set
2. Best model is to be selected on the cross-validation set
3. Model performance is to be evaluated on the test set

#### Tool #1: Validation curve (for hyperparameter tuning)

For a particular hyperparameter, a validation curve can help if the model is overfitting or underfitting as its value varies.

* If an error is high for both the training and the CV sets, the model is underfitting - High bias (Region A).
* If an error is high for the CV set, but low for the training set, the model is overfitting - High variance (Region B).

<img src="images/validation_curve.png" style="zoom:50%"/>

#### Tool #2: Learning curve

A learning curve can help if the model is overfitting or underfitting as the number of training sample varies.

- If errors of the training and the CV sets converges to a high value, the model is underfitting - High bias (Fig. A).
- If an error of the CV set is way higher than that of the training set, the model is overfitting - High variance (Fig. B).

<img src="images/learning_curve.png" style="zoom:50%"/>

#### Some options to consider for improvement

|                  | Consider                                                     |
| ---------------- | ------------------------------------------------------------ |
| **Overfitting**  | <ul><li>Increasing **_m_**</li><li>Decreasing **_n_** by removing some less important features</li><li>More regularization</li></ul> |
| **Underfitting** | <ul><li>Note: increasing **_m_** does not help improving the model performance</li><li>Increasing **_n_** by adding more features</li><li>Less regularization</li></ul> |

1. If a model is overfitting (high variance), consider:
   * Increasing **_m_**
   * Decreasing **_n_** by removing some less important features
   * More regularization
2. If a model is underfitting (high bias), consider:
   * Note: increasing **_m_** does not help improving the model performance
   * Increasing **_n_** by adding more features
   * Less regularization

## For classification model (binary)

The model performance is usually evaluated by various evaluation methods (i.e. accuracy, user satisfaction, survival rate of patient, etc.). **_Accuracy_** is the most popular default choice, but it may give a _partial_ picture of a performance only in many scenarios.

* Typical example: imbalanced class (only a small subset of data explains positive class)
  * Possible scenarios: occurance of fire in a city, occurance of malignant tumors, credit fraud detection, etc.
  * Problem: Even an untrained model (dummy classifier) may give higher accuracy by predicting all as negative 