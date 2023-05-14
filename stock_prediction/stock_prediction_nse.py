import pandas as pd
from sqlalchemy import create_engine
import urllib
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy
import concurrent.futures
import os
from sqlalchemy.exc import InterfaceError, OperationalError


script_config = pd.read_excel("config.xlsx", sheet_name="script_config", engine='openpyxl')
database_config = pd.read_excel("config.xlsx", sheet_name="database_config", engine='openpyxl')

# script config
HISTORY_TABLE = script_config[script_config["Key"] == "HISTORY_TABLE"].Value.item()
RESULTS_TABLE = script_config[script_config["Key"] == "RESULTS_TABLE"].Value.item()
DAYS_TO_USE_TO_TRAIN_MODEL = int(script_config[script_config["Key"] == "DAYS_TO_USE_TO_TRAIN_MODEL"].Value.item())
TIME_STEP = int(script_config[script_config["Key"] == "TIME_STEP"].Value.item())
EPOCH = int(script_config[script_config["Key"] == "EPOCH"].Value.item())
BATCH_SIZE = int(script_config[script_config["Key"] == "BATCH_SIZE"].Value.item())
DAYS_TO_PREDICT = int(script_config[script_config["Key"] == "DAYS_TO_PREDICT"].Value.item())
PREDICTION_COLUMN = script_config[script_config["Key"] == "PREDICTION_COLUMN"].Value.item()

# DB config
SERVER = database_config[database_config["Key"] == "SERVER"].Value.item()
DATABASE = database_config[database_config["Key"] == "DATABASE"].Value.item()

# database connection
try:
    params = urllib.parse.quote_plus("Driver={SQL Server Native Client 11.0};"
                                     f"Server={SERVER};"
                                     f"Database={DATABASE};"
                                     "Trusted_connection=yes")
    engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
    engine.connect()
except InterfaceError:
    engine = create_engine(f'mssql+pyodbc://{SERVER}/{DATABASE}?trusted_connection=yes&driver=ODBC+Driver+11+for+SQL'
                           f'+Server')
    engine.connect()

# for sqlite
# engine = create_engine(r"sqlite:///C:\Users\Anon\PycharmProjects\stock_prediction\stock_prediction.db")
# engine.connect()


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


def get_predictions(data, last_date, model, scaler, days_to_predict=10, n_steps=60):
    x_input = data[(data.shape[0] - n_steps):].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    dates = []
    lst_output = []
    actual_output = []
    i = 0
    d = 0
    while i < days_to_predict:
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            actual_output.append(scaler.inverse_transform(yhat)[0][0])
            d += 1
            dt = last_date + timedelta(d)
            if dt.weekday() > 4:
                if dt.weekday() > 5:
                    d += 1
                    dt = last_date + timedelta(d)
                else:
                    d += 2
                    dt = last_date + timedelta(d)
            dates.append(dt)
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            actual_output.append(scaler.inverse_transform(yhat)[0][0])
            lst_output.extend(yhat.tolist())
            dt = last_date + timedelta(i + 1)
            d += 1
            dt = last_date + timedelta(d)
            if dt.weekday() > 4:
                if dt.weekday() > 5:
                    d += 1
                    dt = last_date + timedelta(d)
                else:
                    d += 2
                    dt = last_date + timedelta(d)
            dates.append(dt)
            i = i + 1
    df = pd.DataFrame(data=[], columns=["date", f"{PREDICTION_COLUMN}_predictions"])
    df["date"] = dates
    df[f"{PREDICTION_COLUMN}_predictions"] = actual_output
    return df


def train_model(QUERY, symb):
    with engine.connect() as con:
        df = pd.read_sql(QUERY.format(symbol=symb), con).sort_values(by="date")
    print("**********************************************************")
    print(f"Training model for {symb} symbol with {df.shape[0]} days")
    df1 = df.reset_index()[PREDICTION_COLUMN]
    if df.shape[0] < DAYS_TO_USE_TO_TRAIN_MODEL:
        print(f"Skipping {symb} due to less data {df.shape[0]}")
        return
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    x_train, y_train = create_dataset(df1, TIME_STEP)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=0)
    preds_df = get_predictions(df1, df.date.iloc[-1], model, scaler, DAYS_TO_PREDICT, TIME_STEP)
    preds_df.insert(loc=1, column="symbol", value=symb)
    preds_df["timestamp"] = datetime.now()
    preds_df = preds_df.round(2)
    with engine.connect() as con:
        preds_df.to_sql(f"{RESULTS_TABLE}", con, if_exists="append", index=False)
    print(f"Done training and predicting {symb}")


def multiprocess_train(function, query, iterable_list):
    if not hasattr(iterable_list, "__iter__"):
        raise Exception("Please pass iterable object")
    try:
        print(f"Total cpu cores in the system are {os.cpu_count()}. Model will be trained on {os.cpu_count()} "
              f"parallel processes.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(executor.map(function, [query]*len(iterable_list), iterable_list))
    except KeyboardInterrupt as e:
        print(e)
    return True


if __name__ == "__main__":
    with engine.connect() as conn:
        try:
            old_result = conn.execute(f"select * from {RESULTS_TABLE}").fetchone()
            if old_result:
                ans = input(f"{RESULTS_TABLE} table already exists. Do you want to drop the table? Y/N")
                if ans.lower() == "y":
                    conn.execution_options(autocommit=True).execute(f"drop table {RESULTS_TABLE}")
        except OperationalError:
            pass

        stock_names = [i.symbol for i in conn.execute(f"select distinct(symbol) from {HISTORY_TABLE}").fetchall()]

    # for ms sql
    QUERY = f"select [date], [{PREDICTION_COLUMN}] * from {HISTORY_TABLE} where symbol='{{symbol}}' order by date desc"
    if DAYS_TO_USE_TO_TRAIN_MODEL:
        QUERY = f"select top({DAYS_TO_USE_TO_TRAIN_MODEL}) [date], [{PREDICTION_COLUMN}] from {HISTORY_TABLE} where " \
                f"symbol='{{symbol}}' order by date desc"

    # for sqlite
    # QUERY = f"select [date], [{PREDICTION_COLUMN}] from {HISTORY_TABLE} where " \
    #         f"symbol='{{symbol}}' order by date desc limit {DAYS_TO_USE_TO_TRAIN_MODEL}"

    multiprocess_train(train_model, QUERY, stock_names)
