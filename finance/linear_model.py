from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from .models import StockData


def prepare_data(symbol):
    # Fetch historical data for the symbol
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')

    # Check if data exists
    if not stock_data.exists():
        return None

    # Create a DataFrame
    data = pd.DataFrame(list(stock_data.values('date', 'close_price')))

    # Convert 'date' to a numerical value (e.g., days since the first date)
    data['days'] = (pd.to_datetime(data['date'])-pd.to_datetime(data['date'].min())).dt.days

    # Prepare feature (X) and target (y) variables
    X = data[['days']]  # The feature is the number of days
    y = data['close_price']  # The target is the close price

    # Split the data into training and testing sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


def train_and_predict(symbol):
    # Prepare the data
    data = prepare_data(symbol)
    if data is None:
        return {"error":"No data available"}

    X_train, X_test, y_train, y_test = data

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Print or return the evaluation metrics
    print(f"Mean squared error: {mse}")
    print(f"R2 score: {r2}")

    return {"predictions":predictions, "mse":mse, "r2":r2}


import pickle


def train_and_save_model(symbol, filename='linear_model.pkl'):
    data = prepare_data(symbol)
    if data is None:
        return {"error":"No data available"}

    X_train, X_test, y_train, y_test = data

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    return model
