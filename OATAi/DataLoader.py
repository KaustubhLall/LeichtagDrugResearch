import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoaderMetabolite:

    def __init__(self, scale=False):
        """
        Sets up a data loader object with the specified parameters.
        :param scale: if true, the scipy.StandardScaler will be used to scale the data.
        :type scale: bool
        """
        self.scale = scale

    def load_oat1_3_small(self):
        """
        Loads the for small fold metabolites.
        :return: X, Y, Header (names of features)
        :rtype:
        """
        source_df = pd.read_csv('./datasets/metabolites/OAT1OAT3Small.csv')
        source_df['SLC'] = source_df['SLC'].astype('category').cat.codes

        to_drop = [0, 2, 3, 4, ]

        df = source_df.drop(source_df.columns[to_drop], axis=1)

        print('Loaded in data, null values found: ', end=' ')
        print(df[pd.isnull(df).any(axis=1)])

        label_index = 1  # this is from source
        print("Data shape: ", df.shape[0])

        X = np.array([np.array(df.iloc[x, :]) for x in range(df.shape[0])])
        Y = np.array(source_df.iloc[:, label_index])

        header = np.array(df.columns)

        if self.scale:
            feature_scaler = StandardScaler()
            X = feature_scaler.transform(X)

        return X, Y, header

    def load_oat1_3_big(self):
        """
        Loads the for large fold metabolites.
        :return: X, Y, Header (names of features)
        :rtype:
        """
        source_df = pd.read_csv('./datasets/metabolites/OAT1OAT3Big.csv')
        source_df['SLC'] = source_df['SLC'].astype('category').cat.codes

        to_drop = [0, 2, 3, 4, ]

        df = source_df.drop(source_df.columns[to_drop], axis=1)

        print('Loaded in data, null values found: ', end=' ')
        print(df[pd.isnull(df).any(axis=1)])

        label_index = 1  # this is from source
        print("Data shape: ", df.shape[0])

        X = np.array([np.array(df.iloc[x, :]) for x in range(df.shape[0])])
        Y = np.array(source_df.iloc[:, label_index])

        header = np.array(df.columns)

        if self.scale:
            feature_scaler = StandardScaler()
            X = feature_scaler.transform(X)

        return X, Y, header

    def load_oat1_3_p_combined(self):
        """
        Loads the for all metabolites, including liver.
        :return: X, Y, Header (names of features)
        :rtype:
        """

        return
        # todo the dataset is missing a header, get the header before proceeding.
        source_df = pd.read_csv('./datasets/metabolites/OAT1OAT3OATP.csv')
        source_df['SLC'] = source_df['SLC'].astype('category').cat.codes

        # todo update this
        to_drop = [0, 1, 3, 4, ]

        df = source_df.drop(source_df.columns[to_drop], axis=1)

        print('Loaded in data, null values found: ', end=' ')
        print(df[pd.isnull(df).any(axis=1)])

        label_index = 2  # this is from source
        print("Data shape: ", df.shape[0])

        X = np.array([np.array(df.iloc[x, :]) for x in range(df.shape[0])])
        Y = np.array(source_df.iloc[:, label_index])

        header = np.array(df.columns)

        if self.scale:
            feature_scaler = StandardScaler()
            X = feature_scaler.transform(X)

        return X, Y, header


dat = DataLoaderMetabolite()

x, y, h = dat.load_oat1_3_big()
# print(dat.load_oat1_3_big())
