import numpy as np


# Randomly split df into a Training & Testing Data
def splitData(df, splitRatio):

    # Make sure that only the first Shuffled values are used throughout the coding process to avoid Overfitting
    np.random.seed(42)
    shuffle = np.random.permutation(len(df))
    testSize = int(len(df) * splitRatio)
    testIndices = shuffle[:testSize]
    trainIndices = shuffle[testSize:]

    # Return a Training df & a Testing df
    return df.iloc[trainIndices], df.iloc[testIndices]
