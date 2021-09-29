import numpy as np
import time

from classification_systems.get_mnist_data import get_data

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

def cross_val_score(clf, data, target, k_folds):
    ''' K-fold cross-validation implementation using only numpy '''

    data = np.concatenate([data, target.reshape(-1, 1)], axis=1)
    splited_data = np.split(data, k_folds)
    for k in range(k_folds):
        clone_clf = clone(clf)
        splited_data_cp = splited_data.copy()
        test_fold = splited_data_cp.pop(k)
        train_folds = np.concatenate(splited_data_cp)
        X_train, X_test = np.delete(train_folds, -1, 1), np.delete(test_fold, -1, 1)
        y_train, y_test = train_folds[:, -1], test_fold[:, -1]
        
        clone_clf.fit(X_train, y_train)
        y_pred = clone_clf.predict(X_test)
        # Accuracy
        n_correct = (y_pred == y_test).sum()
        print(n_correct / len(y_pred))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    sgd_clf = SGDClassifier(random_state=15)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print(confusion_matrix(y_train_5, y_train_pred))
    print(confusion_matrix(y_train_5, y_train_5))
