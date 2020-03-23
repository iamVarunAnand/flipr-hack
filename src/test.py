# import the necessary packages
import pandas as pd
import numpy as np
import pickle
import time

# read the test data
test_df = pd.read_csv("../input/task2_test.csv")
ids = test_df["people_ID"].values
test_df = test_df.drop("people_ID", axis = 1)

mod_df = test_df.copy(deep = True)

categ_cols = [c for c in test_df.columns if c in ["Region", "Gender", "Married",
                                                  "Occupation", "Mode_transport",
                                                  "comorbidity", "Pulmonary score",
                                                  "cardiological pressure"]]

cont_cols = [c for c in test_df.columns if c not in categ_cols]
print(len(cont_cols))

f = open("../output/ohe.pkl", "rb")
ohe = pickle.load(f)
f.close()
transformed_features = ohe.transform(test_df[categ_cols].values)

# drop the categorical columns from the output dataframe
mod_df = mod_df.drop(categ_cols, axis = 1)

# loop through the entries in the transformed features
start_time = time.time()
for i in range(transformed_features.shape[1]):
    col_name = "_".join(["feat", f"{i}"])
    mod_df[col_name] = transformed_features[:, i].todense()

print(f"[INFO] time taken = {time.time() - start_time}")
test_df = mod_df

# loop through the models
preds = []
for fold in range(5):
    # read the current model from disk
    model_path = f"../models/xgboost_{fold}"
    f = open(model_path, "rb")
    model = pickle.load(f)
    f.close()

    # get the current predictions
    cur_preds = model.predict(test_df)

    # add the current predictions to the preds list
    preds.append(cur_preds)

# convert the predictions into a numpy array
preds = np.array(preds)

# take the average of the predictions
preds = np.mean(preds, axis = 0)

# create the output dictionary
sub = {
    "people_ID": ids,
    "Infect_Prob": preds
}

# create the submission file and save it to disk
output_df = pd.DataFrame(sub, columns = ["people_ID", "Infect_Prob"])
output_df.to_csv("../output/submission_task_two.csv", index = False)
