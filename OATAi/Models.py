from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


def train_dt(X, Y, max_depth=7, criterion='entropy', min_samples_leaf=3):
    """
    Trains a decision tree from the provided data.

    :param X:   Data to train decision tree model.
    :param Y:   Labels to train decision tree model.

    :return:    trained model
    """
    train_acc = 0

    # make and fit model
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    clf.fit(X, Y)

    # predict and test model
    train_acc += clf.score(X, Y)

    return clf, train_acc


def train_rfw(X, Y, n_estimators=100, bootstrap=False, criterion='gini'):
    """
    Trains a random forest model from the provided data.

    :param X:   Data to train random forest model.
    :param Y:   Labels to train random forest model.

    :return:    trained model
    """
    train_acc = 0
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, bootstrap=bootstrap, criterion=criterion)
    clf.fit(X, Y)

    # predict and test model
    train_acc += clf.score(X, Y)

    return clf, train_acc


def test_model(model, test_x, test_y):
    """
    Finds the test (validation or hold out) score of any model with a .predict function.

    :param model:   model to test.
    :param test_x:   test data.
    :param test_y:   test labels.

    :return:        test accuracy.
    """

    # find the accuracy (AUC) score
    test_acc = model.score(test_x, test_y)

    # return AUC of validation set
    return test_acc


class ModelXVal:
    def __init__(self):
        self.num_splits = 5

    def dt_xval(self, data, labels, **kwargs):
        kf = KFold(n_splits=self.num_splits, shuffle=True)

        ns = kf.get_n_splits(data)
        acc = 0

        for traini, testi in kf.split(data):
            data_train, data_test = data[traini], data[testi]
            labels_train, labels_test = labels[traini], labels[testi]
            model, _ = train_dt(data_train, labels_train, **kwargs)
            acc += test_model(model, data_test, labels_test)

        score = acc / ns

        return score

    def rf_xval(self, data, labels, **kwargs):
        kf = KFold(n_splits=self.num_splits, shuffle=True)

        ns = kf.get_n_splits(data)
        acc = 0

        for traini, testi in kf.split(data):
            data_train, data_test = data[traini], data[testi]
            labels_train, labels_test = labels[traini], labels[testi]
            model, _ = train_rfw(data_train, labels_train, **kwargs)
            acc += test_model(model, data_test, labels_test)

        score = acc / ns

        return score
