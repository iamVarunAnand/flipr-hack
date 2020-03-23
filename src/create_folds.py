# import the necessary packages
from sklearn.model_selection import KFold
import pandas as pd

if __name__ == "__main__":
    # read the dataset and create an additional folds column
    df = pd.read_csv("../input/mod_train.csv")
    df["fold"] = -1

    # shuffle the dataset
    df = df.sample(frac = 1, random_state = 42).reset_index(drop = True)

    # initialize the kfold splitter
    kf = KFold(n_splits = 5)

    # loop through the folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "fold"] = fold

    # sanity check
    print(df.fold.value_counts())

    # save the dataset to disk
    df.to_csv("../input/mod_train_folds.csv", index = False)
