import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow


def evaluation_function(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


if __name__ == '__main__':
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    data = pd.read_csv(csv_url, sep=";")
    X = data.drop(columns='quality')
    y = data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # ML flow tracking
    with mlflow.start_run():
        alpha = 0.05
        l1_ratio = 0.01
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
        lr.fit(X_train, y_train)

        predictions = lr.predict(X_test)

        (rmse, mae, r2) = evaluation_function(y_test, predictions)

        print(f"rmse s {rmse}")
        print(f"mae is {mae}")
        print(f"r2 is {r2}")

        # Log parameters
        mlflow.log_param("alpha", alpha)  # key value pair
        mlflow.log_param("l1_ratio", l1_ratio)
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        # Log model
        mlflow.sklearn.log_model(lr, "model")  # "model" here is the name of pkl file in which model is saved

        mlflow.end_run()
