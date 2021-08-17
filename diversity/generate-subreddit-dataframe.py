import pandas as pd
import numpy as np


if __name__ == "__main__":
    """
    Before running this file, please ensure that you have generated the vectors.tsv and metadata.tsv files from the subreddit2vec.ipynb file and that they are in the `datasets` folder.
    """

    # Open data
    df = pd.read_csv('datasets/vectors.tsv', sep='\t', header=None)
    df_labels = pd.read_csv('datasets/metadata.tsv', sep='\t', names=['labels'])
    df['vector'] = df[:].values.tolist()
    dfnew = pd.concat([df_labels, df], axis = 1)
    dfnew = dfnew.iloc[37845:47000]
    dfnew.to_csv('datasets/subreddit_embeddings.csv')