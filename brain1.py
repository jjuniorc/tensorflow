# Ref Source: https://medium.com/@pierreia/make-an-ai-powered-bot-for-pancakeswap-prediction-part-1-ddc66819ad91

# pip3 install pandas
import pandas as pd

# pip3 install numpy
import numpy as np

# pip3 install -U scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# pip3 install python-binance
from binance.client import Client

modelX = None
scalerX = None
trained = False
api = {'key':'mLSxbjwzPSlakxzOjqOiKg7cdlRWuyzhEwsrCj7WuxCbzCjrMstBmhIEBid55qtk','secret':'dvlIfurf3ckuVM8y1WEzkqtB6WB83Ty0x79dz9igNGqlitqlQQM4nQDv75l51Z6E'}

def get_prices(window):
    client = Client(api['key'], api['secret'])
    klines = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_1MINUTE, window + " UTC")
    klines_close = [kline[1] for kline in klines]  # la colomne des valeurs de clotures est la colonne 1

    return klines_close
    
def train():
    global modelX
    global scalerX
    global trained
    
    print("Creating model...")
    modelX = RandomForestClassifier(max_depth=20, n_estimators=500)
    print("Scraping prices...")

    closep = get_prices("3 day")
    X_l = []
    Y_l = []

    length = 5
    for i in range(length, len(closep) - length):
        if closep[i] < closep[i + length]:
            Y_l.append(1)
        else:
            Y_l.append(0)
        X_l.append(closep[i - length:i])

    X = np.array(X_l)
    Y = np.array(Y_l)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scalerX = scaler

    print('Training model...')
    modelX.fit(X,Y)
    print("Model trained")
    
    trained = True
    
def predict():
    global modelX
    global scalerX
    global trained
    
    # X = data[id]["last prices"]
    last_prices_brut = get_prices("5 minute")
    while len(last_prices_brut) < 5:
        sleep(1)
        print("Error while scraping prices")
        last_prices_brut = get_prices("5 minute")
    last_prices = scalerX.transform(np.array([last_prices_brut]))

    X = last_prices
    
    prediction = modelX.predict(X)[0]
    if prediction == 0:
        print("Prediction : Down ")
    else:
        print("Prediction : Up ")

train()

predict()