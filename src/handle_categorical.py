# import the necessary packages
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pickle
import time

# read the data
train_df = pd.read_csv("../input/train.csv")
train_df["people_ID"] = -1
test_df = pd.read_csv("../input/test.csv")
test_df["Infect_Prob"] = -1

# concatenate the train and test dataframes
df = pd.concat([train_df, test_df], axis = 0, sort = False)
train_len = len(train_df)

# handle nan values
df = df.fillna("-99999")

# create a copy of the dataframe
output_df = df.copy(deep = True)

# initialize the categorical and continuous columns
categ_cols = [c for c in df.columns if c in ["Region", "Gender", "Married",
                                             "Occupation", "Mode_transport",
                                             "comorbidity", "Pulmonary score",
                                             "cardiological pressure"]]

cont_cols = [c for c in df.columns if c not in categ_cols]

print(len(cont_cols))

# one hot encode the categorical columns
ohe = OneHotEncoder()
transformed_features = ohe.fit_transform(df[categ_cols].values)

# drop the categorical columns from the output dataframe
output_df = output_df.drop(categ_cols, axis = 1)

# loop through the entries in the transformed features
start_time = time.time()
for i in range(transformed_features.shape[1]):
    col_name = "_".join(["feat", f"{i}"])
    output_df[col_name] = transformed_features[:, i].todense()

print(f"[INFO] time taken = {time.time() - start_time}")

# extract the train and test df from the output df
output_train = output_df[:train_len]
output_train = output_train.drop("people_ID", axis = 1)
output_test = output_df[train_len:]
output_test = output_test.drop("Infect_Prob", axis = 1)

# save the modified dataframe to disk
output_train.to_csv("../input/mod_train.csv", index = False)
output_test.to_csv("../input/mod_test.csv", index = False)

# pickle the encoder to disk
f = open("../output/ohe.pkl", "wb+")
pickle.dump(ohe, f)
f.close()
