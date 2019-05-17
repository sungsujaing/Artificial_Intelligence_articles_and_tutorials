# Model evaluation and evalution metrics

During the model building process, understanding the model performance and setting an appropriate single-value evaluation metrics are critical as they can guide how to improve the model performance.

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

* If an error is high for both the training and the CV sets, the model is underfitting - High bias
* If an error is high for the CV set, but low for the training set, the model is overfitting - High variance

<img src="images/validation_curve.png" width="500">

#### Tool #2: Learning curve

A learning curve can help if the model is overfitting or underfitting as the number of training sample varies.

- If errors of the training and the CV sets converges to a high value, the model is underfitting - High bias (Fig. a)
- If an error of the CV set is way higher than that of the training set, the model is overfitting - High variance (Fig. b)

<img src="images/learning_curve.png" swidth="500">

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

* Typical example: imbalanced or skewed class (only a small subset of data explains positive class)
  * Possible scenarios: occurance of fire in a city, occurance of malignant tumors, credit fraud detection, etc.
  * Problem: Even an untrained model (dummy classifier) may give higher accuracy by predicting all as negative 
* Solution: consider other evaluation metrics such as precision and recall (confusion matrix)
  * **_<u>P</u>_**recision = % True among **_<u>P</u>_**redicted as True (important when to avoid False **_<u>P</u>_**ositive)
  * **_<u>R</u>_**ecall = % True among **_<u>R</u>_**eal True (important when to avoid False Negative)

<img src="images/confusion_matrix.png" width="500">

* Precision and Recall trade-off

  | When?                   | Action                           | Precision | Recall   |
  | ----------------------- | -------------------------------- | --------- | -------- |
  | To avoid False Positive | Raise decision threshold (> 0.5) | Increase  | Decrease |
  | To avoid False Negative | Lower decision threshold (< 0.5) | Decrease  | Increase |

* Two metrics can be difficult to compare when choosing the better model, consider:
  * F<sub>1</sub>  = 2(Precision*Recall) / (Precision + Recall)
  * F<sub>&beta;</sub>  = (1+&beta;<sup>2</sup>)(Precision*Recall) / (&beta;<sup>2</sup>Precision + Recall) 
    * &beta; is set high when recall is important (i.e. ~2)
    * &beta; is set low when precision is important (i.e. ~0.5)
  * Receiver operating characteristic (ROC) curve and its area underneath the curve (AUC)

<img src="images/ROC_curve.png" width="500">

## For classification model (multi-clasee)

As an extension of a simple binary case, overall evaluation metrics are averaged across classes. They are different ways of averaging them and, depending on the distribution of classes, they may lead to different results.

* Type of averaging
  * **Macro**-average: each **class** has equal weight. Compute metrics within each class, then average them.
  * **Micro**-average: each **instance** has equal weight. Aggregate the outcome first, then compute metrics. In this case, large classes have more influence.

| When?                                   | Averaging to do |
| --------------------------------------- | --------------- |
| To weight more toward the small classes | Macro           |
| To weight more toward the large classes | Micro           |

* Compare two averaging results:
  * if micro- << macro-, large classes have poor metrics
  * if micro >> macro-, small classes have poor metrics

## For regression model



