from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class XGBBayesianOptimizer:
    def __init__(self, X_train, y_train, X_validation, y_validation):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation

    def xgb_classifier(self, n_estimators, max_depth, reg_alpha, reg_lambda, min_child_weight,
                       gamma, eta):
        params = {'n_estimators': int(n_estimators),
                  'max_depth': int(max_depth),
                  'reg_alpha': float(reg_alpha),
                  'reg_lambda': float(reg_lambda),
                  'min_child_weight': int(min_child_weight),
                  'gamma': float(gamma),
                  'eta': float(eta)}
        clf = XGBClassifier(**params, use_label_encoder=False)

        clf.fit(self.X_train, self.y_train,
                eval_set = [(self.X_validation, self.y_validation)],
                eval_metric='logloss'
                )
        return 1 - min(list(clf.evals_result()['validation_0'].items())[0][1])

    def optimize(self):
        # Bounded region of parameter space
        pbounds = {'n_estimators': (25, 150),
                  'max_depth': (3, 10),
                  'reg_alpha': (0.0, 0.1),
                  'reg_lambda': (0.0, 0.1),
                  'min_child_weight': (2, 5),
                  # 'max_delta_step': (1, 10),  # Helpful with imbalanced problems
                  'gamma': (0, 5),
                  'eta': (0.01, 0.1)
                   }

        optimizer = BayesianOptimization(
            f=self.xgb_classifier,
            pbounds=pbounds,
            random_state=1)

        optimizer.maximize(
            init_points=4,
            n_iter=16)

        eval_result = 1-optimizer.max['target']
        parameters = {'n_estimators': int(optimizer.max['params']['n_estimators']),
                  'max_depth': int(optimizer.max['params']['max_depth']),
                  'reg_alpha': float(optimizer.max['params']['reg_alpha']),
                  'reg_lambda': float(optimizer.max['params']['reg_lambda']),
                  'min_child_weight': int(optimizer.max['params']['min_child_weight']),
                  'gamma': float(optimizer.max['params']['gamma']),
                  'eta': float(optimizer.max['params']['eta'])}

        return eval_result, parameters

def xgb_grid_search(X_train, y_train, X_validation, y_validation):
    # define model
    model = XGBClassifier()
    # Define exhaustive grid search space
    space = {'n_estimators': [150, 300],
                  'max_depth': [3, 6],
                  'reg_alpha': [0.0, 0.1],
                  'reg_lambda': [0.0, 0.1],
                  'min_child_weight': [1]
             }
    # Define search
    fit_params = {"early_stopping_rounds": 20,
                  "eval_metric": "logloss",
                  "eval_set": [[X_validation, y_validation]]}
    search = GridSearchCV(model, space, scoring='neg_log_loss', n_jobs=-1)
    # Execute search
    result = search.fit(X_train, y_train, **fit_params)
    #Summarize result
    return result.best_params_