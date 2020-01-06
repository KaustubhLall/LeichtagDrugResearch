from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


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
        """
        Runs the random forest classifier with the given parameters in kwargs and reports the cross validation accuracy.

        :param data: input data to run.
        :type data: R^2 numpy array.
        :param labels: input labels to run.
        :type labels: R^1 numpy array.
        :param kwargs: Arguments (optional) for the RFW classifier.
        :type kwargs: dict
        :return: average accuracy over all folds in cross validation
        :rtype: float
        """
        kf = KFold(n_splits=self.num_splits, shuffle=True)

        ns = kf.get_n_splits(data)
        acc = 0

        # for each fold in the xval splits
        for train_index, test_index in kf.split(data):
            # get the data

            data_train, data_test = data[train_index], data[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # train the model and test it
            model, _ = train_dt(data_train, labels_train, **kwargs)
            acc += test_model(model, data_test, labels_test)

        score = acc / ns

        return score

    def rf_xval(self, data, labels, **kwargs):
        """
        Runs the random forest classifier with the given parameters in kwargs and reports the cross validation accuracy.

        :param data: input data to run.
        :type data: R^2 numpy array.
        :param labels: input labels to run.
        :type labels: R^1 numpy array.
        :param kwargs: Arguments (optional) for the RFW classifier.
        :type kwargs: dict
        :return: average accuracy over all folds in cross validation
        :rtype: float
        """
        kf = KFold(n_splits=self.num_splits, shuffle=True)

        ns = kf.get_n_splits(data)
        acc = 0

        # for each fold in the xval splits
        for train_index, test_index in kf.split(data):
            # get the data
            data_train, data_test = data[train_index], data[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # train the model and test it
            model, _ = train_rfw(data_train, labels_train, **kwargs)
            acc += test_model(model, data_test, labels_test)

        score = acc / ns

        return score

    def brute_force_k_fold_x_val(self, num_features, data, labels, logfile='default_log_xval', pref=''):
        total_features = len(data[0])
        log = open(logfile, 'w')

        for k in num_features:

            # for every possible n choose k combinations
            seq = list(combinations(list(range(total_features)), k))

            buf = ''

            for pattern in tqdm(seq, desc='Running Big Dataset'):
                f = open(pref + 'xval_res%d' % k, 'a')

                # get a subset of the data
                sub_data = data[:, pattern]

                # run the classifiers using k fold xval to report the score
                res_dt = self.dt_xval(sub_data, labels)
                res_rf = self.rf_xval(sub_data, labels)

                # write results to file
                s = '%s,%s,'
                # todo fetch header.
                s = s % tuple([res_rf, res_dt]) + ','.join(list(header[list(pattern)]))
                s += '\n'
                print(s, file=log)
                f.write(s)
                f.close()
                buf += s
