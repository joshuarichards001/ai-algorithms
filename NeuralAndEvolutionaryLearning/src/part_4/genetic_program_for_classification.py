import pandas as pd
from gplearn.genetic import SymbolicClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

if __name__ == '__main__':
    # creating data structures
    train_set = pd.read_csv("training.txt", sep=" ")
    test_set = pd.read_csv("test.txt", sep=" ")

    x_train = train_set.drop("Target", axis=1)
    y_train = train_set["Target"]
    x_test = test_set.drop("Target", axis=1)
    y_test = test_set["Target"]

    est = SymbolicClassifier(parsimony_coefficient=.01,
                             stopping_criteria=0.01,
                             feature_names=list(x_train.columns.values),
                             random_state=3)

    est.fit(x_train, y_train)

    y_true = y_test
    y_score = est.predict_proba(x_test)[:, 1]

    print("Accuracy:", roc_auc_score(y_true, y_score), "Program:", est._program)
