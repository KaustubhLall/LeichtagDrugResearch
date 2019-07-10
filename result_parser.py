from collections import Counter

import pandas as pd


def get_dataframe_from_file(filepath, num_features):
    filepath = filepath + '%d' % num_features
    # find header for the thing
    # first two columns will be accuracy
    # rest of the columns will be the features individually

    header = ['Random Forest HOO Accuracy', 'Decision Tree HOO Accuracy'] + [
        'Feature %2d' % i for i in range(1, num_features + 1)]

    df = pd.read_csv(filepath, header=None)
    df.dropna(inplace=True)
    df[0] = df[df.columns[0]].astype('float')
    df[1] = df[df.columns[1]].astype('float')
    df.columns = header

    return df


def parse_logfile(filepath, num_features):
    df = get_dataframe_from_file(filepath, num_features)

    # sort and write results to file
    df_rfw_sorted = df.sort_values(
        by='Random Forest HOO Accuracy',
        axis=0,
        ascending=False
    )

    # this function writes the file to disk
    df_rfw_sorted.to_csv(
        'Top RFW Accuracy small %d features of (all) 52.csv' % num_features,
        sep=',',
    )

    # repeat sort and write for decision tree
    df_dt_sorted = df.sort_values(
        by='Decision Tree HOO Accuracy',
        axis=0,
        ascending=False
    )

    # this function will write the file
    df_dt_sorted.to_csv(
        'Top DT Accuracy small %d features of (all) 52.csv' % num_features,
        sep=',',
    )


def get_all_above_thresh(df, index, thresh, outfile, pref=''):
    outfile = pref + '%d_feature_%s' % (len(df.columns) - 2, outfile)
    bool = df[df.columns[index]] > thresh

    filtered_df = df[bool]
    filtered_df.to_csv('%s.csv' % outfile, sep=',')

    counts = get_counts(filtered_df)

    top_5 = counts.most_common(5)
    top_10 = counts.most_common(10)

    all_counts = str(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    s = '''Top 5 : {}
    Top 10 : {}
    
    All counts: {}
    '''.format(top_5, top_10, all_counts)

    with open('%s_counts.txt' % outfile, 'w') as f:
        f.write(s)

    return filtered_df


def get_counts(df):
    l = []
    for e in df.columns[2:]:
        l += list(df[e])

    return Counter(l)


s = ''

pref = 'big_'  # big_ or small_, prefix to save output files

nfs = [3, 4, 5, 6]  # num features to try
res_file_prefix = 'big_xval_res'  # small_res or big_xval_res, prefix to look for result files

for nf in nfs:
    try:
        # get dataframe with num features from results on disk
        df = get_dataframe_from_file(res_file_prefix, nf)

        # find and save the thresholds
        get_all_above_thresh(df, 0, 0.850, 'above85rfw', pref)
        get_all_above_thresh(df, 1, 0.850, 'above85dt', pref)
        get_all_above_thresh(df, 0, 0.825, 'above825rfw', pref)
        get_all_above_thresh(df, 1, 0.825, 'above825dt', pref)
        get_all_above_thresh(df, 0, 0.800, 'above80rfw', pref)
        get_all_above_thresh(df, 1, 0.800, 'above80dt', pref)
        get_all_above_thresh(df, 0, 0.750, 'above75rfw', pref)
        get_all_above_thresh(df, 1, 0.750, 'above75dt', pref)
        get_all_above_thresh(df, 0, 0.700, 'above70rfw', pref)
        get_all_above_thresh(df, 1, 0.700, 'above70dt', pref)

    except Exception as e:
        print('error running pref=', pref, 'with nf=', nf, 'on resfile=', res_file_prefix)
        print(e)
