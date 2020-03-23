# import the necessary packages
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from dispatcher import MODELS
import pandas as pd
import argparse
import pickle
import os

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fold", required = True,
                type = int, help = "fold to use")
ap.add_argument("-m", "--model", required = True, help = "model to be used")
args = vars(ap.parse_args())

# initialize the fold mapping
FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

# read the dataset
df = pd.read_csv("../input/mod_train_folds.csv")

# extract the current folds
train_df = df[df.fold.isin(FOLD_MAPPPING.get(
    args["fold"]))].reset_index(drop = True)
val_df = df[df.fold == args["fold"]].reset_index(drop = True)

# split the dataset into input and target
y_train = train_df["Infect_Prob"]
x_train = train_df.drop(["Infect_Prob", "fold"], axis = 1)
y_test = val_df["Infect_Prob"]
x_test = val_df.drop(["Infect_Prob", "fold"], axis = 1)

# # normalize the input data
# cont_cols = [c for c in x_train.columns if c not in ["Region", "Gender", "Married",
#                                                      "Occupation", "Mode_transport",
#                                                      "comorbidity", "Pulmonary score",
#                                                      "cardiological pressure"]]
# x_train.loc[:, cont_cols] = normalize(x_train[cont_cols])
# x_test.loc[:, cont_cols] = normalize(x_test[cont_cols])

# initialize the model
model = MODELS[args["model"]]

# fit the model
model.fit(x_train, y_train)

# evaluate the model
y_pred = model.predict(x_test)
print("{}".format(mean_squared_error(y_test, y_pred)))

# serialize the model to disk
output_path = os.path.sep.join(
    ["../models", "_".join([args["model"], str(args["fold"])])])

f = open(output_path, "wb+")
pickle.dump(model, f)
f.close()
