import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import os


def load_data():
    """
    Loads training data from student_scores_train.csv, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../working_data/student_scores_train.csv"))
    serialized_data = pickle.dumps(df)
    return serialized_data


def data_preprocessing(data):
    """
    Deserializes data, performs preprocessing (scaling), and returns serialized features and target.

    Args:
        data (bytes): Serialized training data.

    Returns:
        tuple: Serialized (X, y) pair.
    """
    df = pickle.loads(data)
    df = df.dropna()

    # Example: assume the last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Serialize both features and target
    serialized_pair = pickle.dumps((X_scaled, y))
    return serialized_pair


def build_save_model(data, filename):
    """
    Trains a Linear Regression model, saves it, and returns the MSE on training data.

    Args:
        data (bytes): Serialized (X, y) data.
        filename (str): Model filename.

    Returns:
        float: Mean Squared Error (MSE) on training data.
    """
    X, y = pickle.loads(data)

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    return mse


def load_model_elbow(filename, mse):
    """
    Loads a trained regression model and predicts on test data.

    Args:
        filename (str): Model filename.
        mse (float): MSE value passed from previous task (for demonstration).

    Returns:
        list: Predictions on test data.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(output_path, 'rb'))

    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "../working_data/student_scores_test.csv"))
    df_test = df_test.dropna()

    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(df_test)

    predictions = model.predict(X_test_scaled)
    print(f"Training MSE: {mse}")
    print(f"Predictions on test data: {predictions[:5]}")

    return predictions.tolist()
