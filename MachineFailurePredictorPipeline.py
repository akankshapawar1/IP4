import joblib
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from joblib import load
from sklearn.metrics import matthews_corrcoef


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
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(x)
    # https://datascience.stackexchange.com/a/36087/145654
    # hidden_layer_sizes, activation, learning_rate 15 because number of features is 8. Geoff Hinton says number of
    # neurons should be less than twice the input features (chatgpt)
    clf = MLPClassifier(solver='lbfgs', max_iter=1000)
    parameter_space = {
        'hidden_layer_sizes': [(100, 100), (50, 50,), (100,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    mlp = GridSearchCV(clf, param_grid=parameter_space, cv=5, scoring='matthews_corrcoef')
    mlp.fit(X_minmax, y)
    print(f"Best MCC : {mlp.best_score_} using {mlp.best_params_}")

    best_model = mlp.best_estimator_
    joblib.dump(best_model, 'multi_layer_nn.pkl')


def support_vector_machine(train_path):
    pass


def k_nearest_neighbour(train_path):
    pass


def decision_tree(train_path):
    pass


def softmax_regression(train_path):
    pass


def mlp_model(test_path):
    df = pd.read_csv(test_path)
    x = df[
        ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
         'type_H', 'type_L', 'type_M']]
    y = df['Machine failure']
    model_path = 'multi_layer_nn.pkl'
    loaded_model = load(model_path)
    predictions = loaded_model.predict(x.values)

    mcc = matthews_corrcoef(y, predictions)
    print(f'mcc on test with best paras : {mcc:.4f}')


def main():
    """
    # step 3
    path = 'ai4i2020.csv'
    step_3_output = step_3(path)
    # step 4
    step_4_output = step_4(step_3_output)
    # step 5
    step_5_output = step_5(step_4_output)
    # step 6 (train test split)
    step_6_train, step_6_test = step_6_split(step_5_output)
    """
    train_path = 'train.csv'
    test_path = 'test.csv'
    # multi_layer_nn(train_path)
    mlp_model(test_path)
    support_vector_machine(train_path)
    k_nearest_neighbour(train_path)
    decision_tree(train_path)
    softmax_regression(train_path)


if __name__ == "__main__":
    main()
