from sklearn import ensemble
from sklearn import svm

MODELS = {
    "randomforest": ensemble.RandomForestRegressor(n_estimators = 200,
                                                   n_jobs = 3,
                                                   verbose = 2),
    "extratrees": ensemble.ExtraTreesRegressor(n_estimators = 200,
                                               n_jobs = 3,
                                               verbose = 2),
    "xgboost": ensemble.GradientBoostingRegressor(n_estimators = 200,
                                                  learning_rate = 0.01,
                                                  min_samples_split = 2,
                                                  max_depth = 3)
}
