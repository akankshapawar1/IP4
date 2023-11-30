import joblib
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from joblib import load
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def step_3(path):
    df = pd.read_csv(path)
    df_required = df.loc[:,
                  ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                   'Tool wear [min]', 'Machine failure']]
    saved_file_name = 'step_3.csv'
    df_required.to_csv(saved_file_name, index=False)
    return saved_file_name


def step_4(path):
    df = pd.read_csv(path)
    # converting categorical column Type to continuous variable
    df_type_transformed = pd.get_dummies(df, "type", "_", False, columns=['Type'])
    df_type_transformed['type_H'] = df_type_transformed['type_H'].astype(int)
    df_type_transformed['type_L'] = df_type_transformed['type_L'].astype(int)
    df_type_transformed['type_M'] = df_type_transformed['type_M'].astype(int)

    # Standardizing the data in few columns
    df_type_transformed[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                         'Tool wear [min]']] = StandardScaler().fit_transform(df_type_transformed[
                                                                                  ['Air temperature [K]',
                                                                                   'Process temperature [K]',
                                                                                   'Rotational speed [rpm]',
                                                                                   'Torque [Nm]',
                                                                                   'Tool wear [min]']])
    saved_file_name = 'step_4.csv'
    df_type_transformed.to_csv(saved_file_name, index=False)
    return saved_file_name


def step_5(path):
    df = pd.read_csv(path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    rus = RandomUnderSampler(random_state=42, sampling_strategy="majority")
    x_rus, y_rus = rus.fit_resample(x, y)
    balanced = pd.DataFrame(x_rus)
    balanced['Machine failure'] = y_rus
    saved_file_name = 'step_5.csv'
    balanced.to_csv(saved_file_name, index=False)
    return saved_file_name


def step_6_split(path):
    df = pd.read_csv(path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    train = pd.DataFrame(x_train)
    train['Machine failure'] = y_train
    train.to_csv(train_file_name, index=False)
    test = pd.DataFrame(x_test)
    test['Machine failure'] = y_test
    test.to_csv(test_file_name, index=False)
    return train_file_name, test_file_name


def multi_layer_nn(train_path):
    df = pd.read_csv(train_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    # https://datascience.stackexchange.com/a/36087/145654
    clf = MLPClassifier(solver='lbfgs', max_iter=1000)
    parameter_space = {
        'hidden_layer_sizes': [(100, 100), (50, 50,), (100,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    mlp = GridSearchCV(clf, param_grid=parameter_space, cv=5, scoring='matthews_corrcoef')
    mlp.fit(x, y)

    best_model = mlp.best_estimator_
    joblib.dump(best_model, 'multi_layer_nn.pkl')

    return mlp.best_score_, mlp.best_params_


def support_vector_machine(train_path):
    df = pd.read_csv(train_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    svc = SVC()
    parameter_space = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    svc_grid = GridSearchCV(svc, param_grid=parameter_space, cv=5, scoring='matthews_corrcoef')
    svc_grid.fit(x, y)

    best_model = svc_grid.best_estimator_
    joblib.dump(best_model, 'svm.pkl')

    return svc_grid.best_score_, svc_grid.best_params_


def k_nearest_neighbour(train_path):
    # (n_neighbors, p, algorithm
    df = pd.read_csv(train_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    knn = KNeighborsClassifier()
    parameter_space = {
        'n_neighbors': [3, 5, 8],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    knn_grid = GridSearchCV(knn, param_grid=parameter_space, cv=5, scoring='matthews_corrcoef')
    knn_grid.fit(x, y)

    best_model = knn_grid.best_estimator_
    joblib.dump(best_model, 'knn.pkl')

    return knn_grid.best_score_, knn_grid.best_params_


def decision_tree(train_path):
    # criterion, max_depth, ccp_alpha
    df = pd.read_csv(train_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    tree = DecisionTreeClassifier()
    parameter_space = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [3, 5, 10],
        'ccp_alpha': [0.0, 1.0, 2.0]
    }
    tree_grid = GridSearchCV(tree, param_grid=parameter_space, cv=5, scoring='matthews_corrcoef')
    tree_grid.fit(x, y)

    best_model = tree_grid.best_estimator_
    joblib.dump(best_model, 'tree.pkl')

    return tree_grid.best_score_, tree_grid.best_params_


def softmax_regression(train_path):
    # penalty, C, solver
    df = pd.read_csv(train_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    softmax = LogisticRegression()
    parameter_space = [
        {'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2'], 'C': [1.0, 0.01, 0.001]},
        {'solver': ['newton-cg', 'lbfgs', 'sag'], 'penalty': ['l2'], 'C': [1.0, 0.01, 0.001]}
    ]
    softmax_grid = GridSearchCV(softmax, param_grid=parameter_space, cv=5, scoring='matthews_corrcoef')
    softmax_grid.fit(x, y)

    best_model = softmax_grid.best_estimator_
    joblib.dump(best_model, 'softmax.pkl')

    return softmax_grid.best_score_, softmax_grid.best_params_


def mlp_model(test_path):
    df = pd.read_csv(test_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    model_path = 'multi_layer_nn.pkl'
    loaded_model = load(model_path)
    predictions = loaded_model.predict(x)

    mcc = matthews_corrcoef(y, predictions)
    return mcc


def svm_model(test_path):
    df = pd.read_csv(test_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    model_path = 'svm.pkl'
    loaded_model = load(model_path)
    predictions = loaded_model.predict(x)

    mcc = matthews_corrcoef(y, predictions)
    return mcc


def knn_model(test_path):
    df = pd.read_csv(test_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    model_path = 'knn.pkl'
    loaded_model = load(model_path)
    predictions = loaded_model.predict(x)

    mcc = matthews_corrcoef(y, predictions)
    return mcc


def tree_model(test_path):
    df = pd.read_csv(test_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    model_path = 'tree.pkl'
    loaded_model = load(model_path)
    predictions = loaded_model.predict(x)

    mcc = matthews_corrcoef(y, predictions)
    return mcc


def softmax_model(test_path):
    df = pd.read_csv(test_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    model_path = 'softmax.pkl'
    loaded_model = load(model_path)
    predictions = loaded_model.predict(x)

    mcc = matthews_corrcoef(y, predictions)
    return mcc


def print_train_results(mlp_best_score, mlp_best_params, svm_best_score, svm_best_params, knn_best_score, knn_best_params,
                        tree_best_score, tree_best_params, lr_best_score, lr_best_params):
    # Create a PrettyTable object
    table = PrettyTable()

    # Define the column names
    table.field_names = ["ML Trained Model", "Best Set of Parameter Values",
                         "MCC-score on the 5-fold Cross Validation on Training Data (80%)"]

    # Add rows with the data
    table.add_row(["Multi-layer Neural Network", mlp_best_params, mlp_best_score])
    table.add_row(["Support Vector Machine", svm_best_params, svm_best_score])
    table.add_row(["K-Nearest Neighbors", knn_best_params, knn_best_score])
    table.add_row(["Decision Tree", tree_best_params, tree_best_score])
    table.add_row(["Softmax Regression", lr_best_params, lr_best_score])

    # Print the table to the console
    print(table)


def print_train_results2(mlp_mcc, mlp_best_params, svm_mcc, svm_best_params, knn_mcc, knn_best_params, tree_mcc,
                         tree_best_params, soft_mcc, lr_best_params):

    table = PrettyTable()

    table.field_names = ["ML Trained Model", "Best Set of Parameter Values",
                         "MCC-score on the 5-fold Cross Validation on Testing Data (20%)"]

    table.add_row(["Multi-layer Neural Network", mlp_best_params, mlp_mcc])
    table.add_row(["Support Vector Machine", svm_best_params, svm_mcc])
    table.add_row(["K-Nearest Neighbors", knn_best_params, knn_mcc])
    table.add_row(["Decision Tree", tree_best_params, tree_mcc])
    table.add_row(["Softmax Regression", lr_best_params, soft_mcc])

    print(table)


def main():
    # step 3
    path = 'ai4i2020.csv'
    step_3_output = step_3(path)

    # step 4
    step_4_output = step_4(step_3_output)

    # step 5
    step_5_output = step_5(step_4_output)

    # step 6 (train test split)
    train_path, test_path = step_6_split(step_5_output)

    # step 6 model training
    mlp_best_score, mlp_best_params = multi_layer_nn(train_path)
    svm_best_score, svm_best_params = support_vector_machine(train_path)
    knn_best_score, knn_best_params = k_nearest_neighbour(train_path)
    tree_best_score, tree_best_params = decision_tree(train_path)
    lr_best_score, lr_best_params = softmax_regression(train_path)
    print_train_results(mlp_best_score, mlp_best_params, svm_best_score, svm_best_params, knn_best_score,
                        knn_best_params,tree_best_score, tree_best_params, lr_best_score, lr_best_params)

    # step 7 model testing
    mlp_mcc = mlp_model(test_path)
    svm_mcc = svm_model(test_path)
    knn_mcc = knn_model(test_path)
    tree_mcc = tree_model(test_path)
    soft_mcc = softmax_model(test_path)
    print_train_results2(mlp_mcc, mlp_best_params, svm_mcc, svm_best_params, knn_mcc, knn_best_params,
                         tree_mcc, tree_best_params, soft_mcc, lr_best_params)


if __name__ == "__main__":
    main()
