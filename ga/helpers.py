import neat
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from ga import visualize


def load_dataset(small, verbose=True, scale=False):
    """
    Loads in the wine data described from the disk.

    :param small:       the desired dataset.
    :param verbose:     if true, print summary of data.
    :param scale:     if true, scale the data.

    :return:        X, Y, trainX, trainY, testX, testY
    """

    if small or small == 'small':
        datapath = '../datasets/OAT1-3 Small.csv'
    else:
        datapath = '../datasets/OAT1-3 Big.csv'

    source_df = pd.read_csv(datapath)
    source_df['SLC'] = source_df['SLC'].astype('category').cat.codes

    to_drop = [0, 1, 2, 3, 4, 5, 6, 7]

    df = source_df.drop(source_df.columns[to_drop], axis=1)

    # print(df[pd.isnull(df).any(axis=1)])

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
        X = feature_scaler.fit(X, Y)
        X = feature_scaler.transform(X)

    return X, Y, header


def create_config(conf_file):
    # have everything set to default settings for now, can technically change the config file.
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        conf_file
        )


def run(epochs, fname=None):
    if fname is None:
        conf_filepath = 'base_config'
    else:
        conf_filepath = fname

    # make a config file
    conf = create_config(conf_filepath)

    # make a new population
    pop = neat.Population(conf)

    if fname is not None:
        pop = neat.Checkpointer.restore_checkpoint(fname)

    # make statistical reporters
    stats = neat.StatisticsReporter()
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)

    # make a checkpointer to save progress every 10 epochs
    pop.add_reporter(neat.Checkpointer(100))

    # find the winner
    winner = pop.run(fitness, epochs)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    node_names = headers
    temp = node_names.split(', ')

    node_names = {}
    print(temp)
    for i, e in enumerate(temp):
        node_names[-(i + 1)] = e

    node_names[0] = 'Output'
    visualize.draw_net(conf, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


X, Y, headers = load_dataset('small')


def fitness(genomes, conf):
    """
    Runs a population of genomes.
    :param genomes:
    :param conf:
    :return:
    """
    train_input, train_output = X, Y
    for gid, genome in genomes:
        genome.fitness = len(train_input)
        net = neat.nn.FeedForwardNetwork.create(genome, conf)

        for xi, xo in zip(train_input, train_output):
            pred = net.activate(xi)
            genome.fitness -= sd(pred[0], xo)


def sd(a, b):
    """
    Squared distance of two vectors.
    :param a:
    :param b:
    :return:
    """
    if isinstance(a, list) and isinstance(b, list):
        return msd_list(a, b)

    return (a - b) ** 2


def msd_list(a, b):
    """
    Helper to find squared distance for two vectors.
    :param a:
    :param b:
    :return:
    """
    assert len(a) == len(b), 'Target output length doesnt match predictions'
    return len(a) / sum([sd(*x) for x in zip(a, b)])


run(10000)
