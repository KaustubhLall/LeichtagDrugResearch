import itertools
from copy import deepcopy
from itertools import combinations

# import keras
import numpy as np
import pandas as pd
# from keras import backend as K
# from keras.layers import Dense, Dropout, BatchNormalization, Activation
# from keras.models import Sequential
# from keras.regularizers import l2
# from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm as tqdm


def nck(n, k):
    import operator as op
    from functools import reduce

    def ncr(n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer / denom


def load_dataset(small, verbose=True, scale=True):
    """
    Loads in the wine data described from the disk.
    
    :param small:       the desired dataset. 
    :param verbose:     if true, print summary of data. 
    :param scale:     if true, scale the data. 
    
    :return:        X, Y, trainX, trainY, testX, testY
    """

    if small or small == 'small':
        datapath = 'datasets/OAT1-3 Small.csv'
    else:
        datapath = 'datasets/OAT1-3 Big.csv'

    source_df = pd.read_csv(datapath)
    source_df['SLC'] = source_df['SLC'].astype('category').cat.codes

    to_drop = [0, 1, 2, 3, 4, 5, 6, 7]

    df = source_df.drop(source_df.columns[to_drop], axis=1)

    print(df[pd.isnull(df).any(axis=1)])

    label_index = 1  # this is from source
    print("Loaded in data, number of points =", df.shape[0])

    X = np.array([np.array(df.iloc[x, :]) for x in range(df.shape[0])])
    Y = np.array(source_df.iloc[:, label_index])

    header = np.array(df.columns)

    # print summary   
    if verbose:
        print('''
            Data Shape    : %s
            Label Shape     : %s
        ''' % (X.shape, Y.shape)
              )

    if scale:
        feature_scaler = StandardScaler()
        X = feature_scaler.transform(X)

    return X, Y, header


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


#
# def train_nn(X, Y, testX, testY, n_outputs=2, n_epochs=32, lr=1e-4, plot=True):
#     # change labels to one hot encoding
#     temp_Y = np.zeros((len(Y), 2))
#
#     for i, y in enumerate(Y):
#         temp_Y[i][int(y) - 1] = 1
#
#     Y = temp_Y
#
#     temp_Y = np.zeros((len(testY), 2))
#
#     for i, testY in enumerate(testY):
#         temp_Y[i][int(testY) - 1] = 1
#
#     testY = temp_Y
#
#     num_features = len(X[0])
#     num_labels = n_outputs
#     batch_size = 4
#
#     model = Sequential()
#
#     model.add(Dense(num_features))
#     model.add(Dense(256, kernel_regularizer=l2(0.01)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(32, kernel_regularizer=l2(0.01)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#
#     model.add(Dense(num_labels))
#     model.add(Activation('softmax'))
#     opt = keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#
#     history = model.fit(X, Y, verbose=0, epochs=n_epochs, shuffle=True, batch_size=batch_size, validation_data=
#     (testX, testY))
#     scores = model.evaluate(X, Y)
#     score = "\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100)
#
#     if plot:
#         # summarize history for accuracy
#         plt.plot(history.history['acc'])
#         plt.plot(history.history['val_acc'])
#         plt.title('model accuracy')
#         plt.ylabel('accuracy')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()
#         # summarize history for loss
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()
#
#     print(int(np.sum([K.count_params(p) for p in set(model.trainable_weights)])))
#
#     return model, scores[1]
#

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


def test_model(model, testX, testY):
    """
    Finds the test (validation or hold out) score of any model with a .predict function.

    :param model:   model to test.
    :param testX:   test data.
    :param testY:   test labels.

    :return:        test accuracy.
    """

    # find the accuracy (AUC) score
    test_acc = model.score(testX, testY)

    # return AUC of validation set
    return test_acc


### Miscl helper methods, self explanatory ones are not documented
def dropcols(data, cols):
    """
    Drop columns from R^2 matrix as np array.
    :param data: 
    :param cols: 
    :return: 
    """
    data = deepcopy(data)
    assert isinstance(cols, list)
    keep = list(range(len(data[0])))

    for e in cols:
        keep.remove(e)

    return data[:, keep]


def dt_lo(data, labels, **kwargs):
    lo = LeaveOneOut()

    ns = lo.get_n_splits(data)
    acc = 0

    for traini, testi in lo.split(data):
        data_train, data_test = data[traini], data[testi]
        labels_train, labels_test = labels[traini], labels[testi]
        model, _ = train_dt(data_train, labels_train, **kwargs)
        acc += test_model(model, data_test, labels_test)

    score = acc / ns

    return score


def rf_lo(data, labels, **kwargs):
    lo = LeaveOneOut()

    ns = lo.get_n_splits(data)
    acc = 0

    for traini, testi in lo.split(data):
        data_train, data_test = data[traini], data[testi]
        labels_train, labels_test = labels[traini], labels[testi]
        model, _ = train_rfw(data_train, labels_train, **kwargs)
        acc += test_model(model, data_test, labels_test)

    score = acc / ns

    return score


def dt_xval(data, labels, **kwargs):
    kf = KFold(n_splits=5, shuffle=True)

    ns = kf.get_n_splits(data)
    acc = 0

    for traini, testi in kf.split(data):
        data_train, data_test = data[traini], data[testi]
        labels_train, labels_test = labels[traini], labels[testi]
        model, _ = train_dt(data_train, labels_train, **kwargs)
        acc += test_model(model, data_test, labels_test)

    score = acc / ns

    return score


def rf_xval(data, labels, **kwargs):
    kf = KFold(n_splits=5, shuffle=True)

    ns = kf.get_n_splits(data)
    acc = 0

    for traini, testi in kf.split(data):
        data_train, data_test = data[traini], data[testi]
        labels_train, labels_test = labels[traini], labels[testi]
        model, _ = train_rfw(data_train, labels_train, **kwargs)
        acc += test_model(model, data_test, labels_test)

    score = acc / ns

    return score


def brute_force_leave_one_out(num_features, data, labels, logfile='default_log_hoo', pref=''):
    total_features = len(data[0])  # total number of features in the dataset
    log = open(logfile, 'w')
    for k in num_features:
        f = open(pref + 'hoo_res%d' % k, 'w')

        # for every possibly n choose k combinations
        seq = list(combinations(list(range(total_features)), k))

        buf = ''
        # test the pattern of classifiers
        for pattern in tqdm(seq, desc='Running Small Dataset'):
            # get a subset of the data
            sub_data = data[:, pattern]

            # run the classifiers using hold one out to report score
            res_dt = dt_lo(sub_data, labels)
            res_rf = rf_lo(sub_data, labels)

            # write results to file
            s = '%s,%s,'
            s = s % tuple([res_rf, res_dt]) + ','.join(list(header[list(pattern)]))
            s += '\n'
            # print(s, file=log)
            # f.write(s)

            buf += s

        log.write(buf)
        f.write(buf)


def brute_force_k_fold_x_val(num_features, data, labels, logfile='default_log_xval', pref=''):
    total_features = len(data[0])
    log = open(logfile, 'w')

    for k in num_features:
        f = open(pref + 'xval_res%d' % k, 'w')

        # for every possible n choose k combinations
        seq = list(combinations(list(range(total_features)), k))

        buf = ''
        for pattern in tqdm(seq, desc='Running Big Dataset'):
            # get a subset of the data
            sub_data = data[:, pattern]

            # run the classifiers using k fold xval to report the score
            res_dt = dt_xval(sub_data, labels)
            res_rf = rf_xval(sub_data, labels)

            # write results to file
            s = '%s,%s,'
            s = s % tuple([res_rf, res_dt]) + ','.join(list(header[list(pattern)]))
            s += '\n'
            # print(s, file=log)
            # f.write(s)

            buf += s

        log.write(buf)
        f.write(buf)


# run small dataset
X, Y, header = load_dataset('small', scale=False)
# brute_force_leave_one_out([4], X, Y, 'log_small', 'small_')

# repeat for large dataset
# X, Y, header = load_dataset('big', scale=False)
brute_force_k_fold_x_val([3], X, Y, 'log_big', 'big_')
