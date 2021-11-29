import flask
from flask import request
import pandas as pd
import numpy as np
import pickle
from pickle import dump
from xgboost import XGBClassifier
from processing.myconfig import categorical_cols
from datetime import datetime
from processing.metrics import pr_metrics, f_scores
from processing import optimization
import logging
import re

app = flask.Flask(__name__)
app.config["DEBUG"] = True

logging.basicConfig(filename='record.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@app.route("/retrain", endpoint='retrain', methods=["GET", "POST"])
def retrain_func():
    app.logger.info('Info level log')
    app.logger.warning('Warning level log')

    if request.method == 'GET':
        return '''<h1>GET method example /TEST/</h1>
            <p>Get method. /test/</p>'''
    if request.method == 'POST':
        parameters = request.get_json(silent=True)
        grid_search = parameters['data']['grid_search']
        bayesian_optimization = parameters['data']['bayesian_optimization']
        # parameter_dict = parameters['parameters']

        raw_df = pd.read_csv('train.csv')

        raw_df['Vehicle_Age'] = raw_df['Vehicle_Age'].str.strip('>')
        raw_df['Vehicle_Age'] = raw_df['Vehicle_Age'].str.strip('<')

        train_split = len(raw_df)*0.6
        train_split = int(train_split)
        test_split = len(raw_df)*0.8
        test_split = int(test_split)

        train_df = raw_df.iloc[:train_split, 1:]
        validation_df = raw_df.iloc[train_split:, 1:]
        test_df = raw_df.iloc[test_split:, 1:]

        cat_cols = categorical_cols
        train_df[cat_cols] = train_df[cat_cols].astype("category")
        validation_df[cat_cols] = validation_df[cat_cols].astype("category")
        test_df[cat_cols] = test_df[cat_cols].astype("category")

        train_df = pd.get_dummies(train_df, columns=cat_cols)
        validation_df = pd.get_dummies(validation_df, columns=cat_cols)
        test_df = pd.get_dummies(test_df, columns=cat_cols)

        model_columns = list(train_df.columns)
        remove_cols = ['Response', 'Year_Month']
        for i in remove_cols:
            model_columns.remove(i)
        # model_columns.remove("Response")

        for col in model_columns:
            if col not in validation_df.columns:
                validation_df[f'{col}'] = 0
            if col not in test_df.columns:
                test_df[f'{col}'] = 0

        y_train = train_df[["Response"]]
        X_train = train_df.drop(columns=['Response', 'Year_Month'])
        y_validation = validation_df[['Response']]
        X_validation = validation_df.drop(columns=['Response', 'Year_Month'])
        y_test = test_df[['Response']]
        X_test = test_df.drop(columns=['Response', 'Year_Month'])

        if bayesian_optimization:
            opt = optimization.XGBBayesianOptimizer(X_train, y_train.values, X_validation[model_columns], y_validation.values)
            eval_result, parameter_dict = opt.optimize()
        elif grid_search:
            parameter_dict = optimization.xgb_grid_search(X_train, y_train, X_validation[model_columns], y_validation)

        model = XGBClassifier(**parameter_dict)
        model.fit(X_train, y_train.values.ravel(),
                  eval_set=[(X_train, y_train.values.ravel()), (X_validation[model_columns], y_validation.values.ravel())],
                  eval_metric='logloss')

        temp_year_month = datetime.today().strftime("%Y-%m")
        dump(model, open(f'health_ins_prediction_model_{temp_year_month}.pkl', 'wb'))
        dump(model_columns, open(f'health_ins_prediction_columns_{temp_year_month}.pkl', 'wb'))

        predicted_probabilities = model.predict_proba(X_test[model_columns])[:, 1]
        f1_f_score, f1_threshold, f05_f_score, f05_threshold = f_scores(predicted_probabilities, y_test.values)

        f05_precision, f05_recall = pr_metrics(predicted_probabilities, f05_threshold, y_test.values)
        f1_precision, f1_recall = pr_metrics(predicted_probabilities, f1_threshold, y_test.values)

        model_performance = {'f1_precision': f1_precision, 'f1_recall': f1_recall,
                             'f05_precision': f05_precision, 'f05_recall': f05_recall,
                             'f1_f_score': f1_f_score, 'f1_threshold': f1_threshold,
                             'f05_f_score': f05_f_score, 'f05_threshold': f05_threshold}
        dump(model_performance, open(f'health_insurance_model_performance_{temp_year_month}.pkl', 'wb'))

        return model_performance


@app.route("/predict", endpoint='predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return '''<h1>GET method example /TEST/</h1>
                    <p>Get method. /test/</p>'''
    if request.method == 'POST':
        data = request.get_json(silent=True)

        df = pd.DataFrame.from_dict(data['data'])

        column_names = categorical_cols
        df[column_names] = df[column_names].astype("category")
        df = pd.get_dummies(df, columns=column_names)

        df['pred'] = "false"
        df['pred_probability'] = 0

        temp_year_month = df['Year_Month'].unique()
        for m in temp_year_month:
            try:
                model_columns = pickle.load(open(f'health_ins_prediction_columns_{m}.pkl', 'rb'))
                model = pickle.load(open(f'health_ins_prediction_model_{m}.pkl', 'rb'))
                model_performance = pickle.load(open(f'health_insurance_model_performance_{m}.pkl', 'rb'))

                for col in model_columns:
                    if col not in df.columns:
                        df[f'{col}'] = None

                df[model_columns] = df[model_columns].astype(float)
                df['pred_probability'] = np.where(df['Year_Month'] == temp_year_month, model.predict_proba(df[model_columns])[:, 1],
                                                  df['pred_probability'])
                df['pred'] = np.where(df['Year_Month'] == temp_year_month,
                                      df['pred_probability'] >= model_performance['f1_threshold'], df['pred'])
            except:
                df['pred'] = np.where(df['Year_Month'] == temp_year_month, 'Model not found', df['pred'])
                df['pred_probability'] = np.where(df['Year_Month'] == temp_year_month, 'Model not found', df['pred_probability'])

        return df[['id', 'pred', 'pred_probability']].to_json()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)