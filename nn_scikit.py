from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import util

def main(file_path, save_path):
    """
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    X, Y = util.load_dataset(file_path, add_intercept=True)
    x_train, y_train, x_eval, y_eval = train_test_split(X, Y, random_state=1)

    # activation: use either 'logistic' (sigmoid), or 'relu'
    # hidden layer size: 2 layers each with 100 units
    nn = MLPRegressor(max_iter=300, activation="relu", hidden_layer_sizes= (100, 100))

    nn.fit(x_train, y_train)
    pred = nn.predict(x_eval)

    print("MSE: ", mean_squared_error(y_eval, pred))
    print("MAE: ", mean_absolute_error(y_eval, pred))

    np.savetxt(save_path, pred)

    # test commit message (disregard)

if __name__ == '__main__':
    # edit
    main(file_path='cleaned_data.csv',
         save_path='logreg_pred.txt')