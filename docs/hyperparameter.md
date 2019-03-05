# petfinder  - Hyperparameter Tuning
## by Robin Brecht and Philipp Paraguya

Here we are logging our hyperparameter experiments. Doing a brute force hyperparameter tuning (via i.e. GridSearch) is currently not possible because lack of machinery capable of doing so.

Algorithm | Parameters | Accuracy (Local) | Accuracy (Kaggle) 
--- | --- | --- | ---
LGBM | objective='multiclass', num_leaves= 70, max_depth = 9, learning_rate = 0.01, lambda_l2 = 0.0475, bagging_fraction= 0.85 | 0.315 | 0.288
