import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import time
import datetime

import warnings

warnings.filterwarnings(action='ignore')


# function for find the best accuracy given datas, params
def find_best(X, y, scalers, models, k_fold_num=10, params_dict=None,
              cv_list=[10]):
    # save the best accuracy each models
    best_accuracy = {}

    # find the best parameter by using grid search
    for scaler_key, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        print(f'------scaler: {scaler_key}------')
        for model_key, model in models.items():
            print(f'----model: {model_key}----')
            for train_idx, test_idx in KFold(n_splits=k_fold_num).split(X):
                start_time = time.time()  # for check running time

                # train test split
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # grid search
                grid = GridSearchCV(model, param_grid=params_dict[model_key])
                grid.fit(X_train, y_train)
                print(f'params: {grid.best_params_}')
                best_model = grid.best_estimator_
                predict = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, predict)

                # save the 5 highest accuracy and parameters each models
                save_len = 7
                save_len -= 1
                flag = False

                target_dict = {'accuracy': accuracy, 'scaler': scaler_key,
                               'model': model_key, 'k_fold_num': k_fold_num,
                               'param': grid.best_params_}
                # save accuracy if best_accuracy has less than save_len items
                if model_key not in best_accuracy.keys():
                    best_accuracy[model_key] = []
                if len(best_accuracy[model_key]) <= save_len:
                    best_accuracy[model_key].append(target_dict)
                # insert accuracy for descending
                elif best_accuracy[model_key][-1]['accuracy'] < accuracy:
                    for i in range(1, save_len):
                        if best_accuracy[model_key][save_len - 1 - i]['accuracy'] > accuracy:
                            best_accuracy[model_key].insert(save_len - i, target_dict)
                            best_accuracy[model_key].pop()
                            flag = True
                            break
                    if flag is False:
                        best_accuracy[model_key].insert(0, target_dict)
                        best_accuracy[model_key].pop()

                print(f'accuracy: {accuracy}', end='')
                end_time = time.time()  # for check running time
                print(f'   running time: {end_time - start_time}  cur_time: {datetime.datetime.now()}', end='\n\n')

    print(f'------train result------')
    displayResultDict(best_accuracy)

    best_score = crossValidation(X, y, scalers, models, best_accuracy, cv_list)

    return best_score


def preprocessing():
    df = pd.read_csv(r'.\dataset\breast-cancer-wisconsin.data', header=None)

    ## Preprocessing

    # # check outlier
    # print(df.iloc[:, 1:].info())
    # print(df.iloc[:, 6].value_counts())
    df.replace('?', np.NAN, inplace=True)
    df.iloc[:, 6] = pd.to_numeric(df.iloc[:, 6])

    # # check NAN
    # print(df.isna().sum().sum())

    df_drop_NAN = df.dropna(axis=0)

    # print(df_drop_NAN.iloc[:, 1:].describe())

    id = df_drop_NAN.iloc[:, 0].copy()
    X = df_drop_NAN.iloc[:, 1:-2].copy()
    y = df_drop_NAN.iloc[:, -1].copy()

    # print(y.value_counts())

    y.replace(2, 0, inplace=True)
    y.replace(4, 1, inplace=True)

    # # check target ratio
    # print(y.value_counts())

    return X, y


# function for set hyper parameters and run find_best
def train():
    X, y = preprocessing()

    # 1. scaler : Standard, MinMax, Robust

    standard = StandardScaler()
    minMax = MinMaxScaler()
    robust = RobustScaler()

    # 2. Model : Decision tree(entropy), Decision tree(Gini), Logistic regression, SVM

    decisionTree_entropy = tree.DecisionTreeClassifier(criterion="entropy")
    decisionTree_gini = tree.DecisionTreeClassifier(criterion="gini")
    logistic = LogisticRegression()
    svm_model = svm.SVC()

    # save scalers and models and hyper parameters in dictionary

    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust}
    models = {"decisionTree_entropy": decisionTree_entropy, "decisionTree_gini": decisionTree_gini,
              "logistic": logistic, "svm": svm_model}

    params_dict = {"decisionTree_entropy": {"max_depth": [x for x in range(3, 11, 1)],
                                            "min_samples_split": [x / 101 for x in range(2, 50, 5)] + [2, 4, 8, 16],
                                            "min_samples_leaf": [x for x in range(5, 10, 1)]},
                   "decisionTree_gini": {"max_depth": [x for x in range(3, 11, 1)],
                                         "min_samples_split": [x / 101 for x in range(2, 50, 5)] + [2, 4, 8, 16],
                                         "min_samples_leaf": [x for x in range(5, 10, 1)]},
                   "logistic": {"C": [1, 2, 3, 5], "tol": [1e-1, 1, 5, 10],
                                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                'penalty': ['l1', 'l2', 'elasticnet', 'none']},
                   "svm": {"C": [1, 2, 3, 4, 5], "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                           "gamma": ['scale', 'auto'], "tol": [1e-4, 1e-1, 1]}
                   }

    # kFold's n_splits parameter
    k_fold_num = 10
    cv_list = [5, 7, 10]

    best_score = find_best(X, y, scalers, models, k_fold_num, params_dict, cv_list)

    return best_score


def crossValidation(X, y, scalers, models, result_dict, cv_list):
    best_score = {}
    for i in models.keys():
        best_score[i] = {'best_score': 0}

    for model_name, result_list in result_dict.items():
        for result in result_list:
            for cv in cv_list:
                scaler_name = result['scaler']
                X_scaled = scalers[scaler_name].fit_transform(X)
                models[model_name].set_params(**result['param'])
                scores = cross_val_score(models[model_name], X_scaled, y, cv=cv)
                result['cv'] = cv
                score_mean = scores.mean()

                if best_score[model_name]['best_score'] < score_mean:
                    best_score[model_name]['best_score'] = score_mean
                    best_score[model_name]['info'] = result

    print(f'------best score------')

    for m in models.keys():
        print(m)
        print(best_score[m])

    return best_score


# function for display result_dict
def displayResultDict(result_dict):
    print(result_dict)
    for model_name, result_list in result_dict.items():
        print(model_name)
        for result in result_list:
            print(result)


if __name__ == "__main__":
    train()