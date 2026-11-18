#!/usr/bin/env python3
"""
Train and export EV Clock model (SVR-RBF)
"""

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

input_file = "InputDataset.xlsx"
data = pd.read_excel(input_file, sheet_name="Data", header=1)

X = data.iloc[:, 4:41]
y = pd.to_numeric(data.iloc[:, 1]) 

preprocessing_and_svr = Pipeline([
    ("log", FunctionTransformer(np.log1p)),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90, svd_solver="full")),
    ("svr", SVR(kernel="rbf")),
])
model = TransformedTargetRegressor(
    regressor=preprocessing_and_svr,
    transformer=StandardScaler(),
)

parameters = {
    "regressor__svr__C": [0.1, 1, 3, 10, 30, 100],
    "regressor__svr__gamma": np.array([0.1, 0.3, 1, 3, 10]) / X.shape[1],
    "regressor__svr__epsilon": [0.01, 0.025, 0.05, 0.10, 0.20],
}
search = GridSearchCV(
    model,
    parameters,
    scoring="neg_mean_absolute_error",
    cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=67),
    refit=True,
)
search.fit(X, y)

print(f"Best CV MAE: {-search.best_score_:.3f} years")
print(f"C: {search.best_params_['regressor__svr__C']:g}")
print(f"Gamma level: {search.best_params_['regressor__svr__gamma'] * X.shape[1]:g}")
print(f"Epsilon: {search.best_params_['regressor__svr__epsilon']:g}")