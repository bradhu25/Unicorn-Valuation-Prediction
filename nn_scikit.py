from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics  import f1_score,accuracy_score
import numpy as np
import util

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler  

# PERCEPTRON IS SENSITIVE TO SCALE
# scaler = StandardScaler()  
#  # Don't cheat - fit only on training data
# scaler.fit(X_train)  
# X_train = scaler.transform(X_train)  
# # apply same transformation to test data
# X_test = scaler.transform(X_test)  

def uni_regression(file_path, model, save_path):
    X, Y = util.load_dataset(file_path, label_col='Last Known Valuation', add_intercept=True)
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, random_state=1)

    # activation: use either 'logistic' (sigmoid), or 'relu'
    # hidden layer size: 2 layers each with 100 units
    if model == "MLPRegressor":
        nn = MLPRegressor(max_iter=300, activation="relu", hidden_layer_sizes= (100, 100))

    nn.fit(x_train, y_train)
    pred = nn.predict(x_eval)

    print(model)
    print("MSE: ", mean_squared_error(y_eval, pred))
    print("MAE: ", mean_absolute_error(y_eval, pred))
    print("\n")

    np.savetxt(save_path, pred)
    

def uni_classification(file_path, model, save_path):
    X, Y = util.load_dataset(file_path, label_col='Unicorn Status', add_intercept=True)
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, random_state=1)

    if model == "Logistic Regression":
        nn = LogisticRegression()
    elif model == "SVC":
        nn = SVC()
    elif model == "SGDClassifier":
        nn = SGDClassifier()

    nn.fit(x_train, y_train)
    pred = nn.predict(x_eval)

    print(model)
    print("Accuracy Score: ", accuracy_score(y_eval,pred))
    print("F1 Score: ", f1_score(y_eval,pred))
    print("\n")

    np.savetxt(save_path, pred)

def main(file_path, save_path):
    """
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    methods = {"Regression": ["MLPRegressor"], "Classification": ["Logistic Regression", "SVC", "SGDClassifier"]}

    for method in methods:
        models = methods[method]
        
        if method == "Regression":
            for model in models:
                uni_regression(file_path, model, model + "_" + save_path)

        if method == "Classification":
            for model in models:
                uni_classification(file_path, model, model + "_" + save_path)

if __name__ == '__main__':
    # edit
    main(file_path='cleaned_data.csv',
         save_path='pred.txt')