# Machine-Learning

## Performance Measures

#### Accuracy
Accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (i.e., when some
classes are much more frequent than others).

#### Confusion Matrix
A much better way to evaluate the performance of a classifier is to look at the confusion matrix. The general idea is to count the number of times instances of class A are classified as class B.

#### Precision and Recall

```math
precision = \frac{TP}{TP + FP}
```