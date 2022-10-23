# Ref Source: https://medium.com/@pierreia/make-an-ai-powered-bot-for-pancakeswap-prediction-part-1-ddc66819ad91

# pip3 install pandas
import pandas as pd

# pip3 install numpy
import numpy as np

# pip3 install -U scikit-learn
from sklearn.preprocessing import StandardScaler

# pip3 install python-binance
from binance.client import Client

# pip3 install matplotlib
import matplotlib.pyplot as plt


# Block
api = {'key':'mLSxbjwzPSlakxzOjqOiKg7cdlRWuyzhEwsrCj7WuxCbzCjrMstBmhIEBid55qtk','secret':'dvlIfurf3ckuVM8y1WEzkqtB6WB83Ty0x79dz9igNGqlitqlQQM4nQDv75l51Z6E'}

# Block
lenght_data = "7 day" #We get 1 minute data for the last 7 days
client = Client(api['key'], api['secret'])
klines = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_1MINUTE, lenght_data + " UTC")
klines_close = [kline[1] for kline in klines]  # close value is column 1

# Block
X_l = []
Y_l = []
lookback = 10 #our lookback is 10 last prices

for i in range(lookback,len(klines_close)-lookback):
    if klines_close[i] < klines_close[i+5]:
        Y_l.append(1)
    else:
        Y_l.append(0)
    X_l.append(klines_close[i-lookback:i])


# Block
X = np.array(X_l) #We convert our data into numpy arrays
Y = np.array(Y_l)

# Block
scaler = StandardScaler()
scaler.fit(X)

# Block
t = int(len(X)*0.8) #we keep 80% of the data for the training
X_train, X_test = scaler.transform(X[:t]), scaler.transform(X[t:])
Y_train, Y_test = Y[:t].copy(), Y[t:].copy()

# Block
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Block
log_rfc = RandomForestClassifier(max_depth=20, n_estimators=500)
log_svm = SVC(probability=True)
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),n_estimators=1000)
gbrc = GradientBoostingClassifier(max_depth=20, n_estimators=1000)

# Block
model = log_rfc

# Block
model.fit(X_train,Y_train)
Y_train_pred = model.predict(X_train)
Y_pred = model.predict(X_test)

# Block
print("Accuracy: " + str(accuracy_score(Y_pred,Y_test)))
#scores = cross_val_score(model,X_train,Y_train,cv=10)

# Block
b = 1 #Start balance for RFC Strategy
bet = 0.1 #Amount of bet
fee = 0.001 #Transaction fees
br = 1 #Start balance for Random Strategy

# Block
l_b = []
l_br = []
for i in range(len(Y_test)):
    odd = np.random.randint(1200,3000)/1000
    rand = np.random.randint(0,2)
    if Y_pred[i] == Y_test[i]:
        b += (odd-1)*bet - 2*fee
    else:
        b -= bet - fee

    if 0 == Y_test[i]:
        br += (odd-1)*bet - 2*fee
    else:
        br -= 1 - fee
    l_b.append(b)
    l_br.append(br)

# Block
rfc_line, = plt.plot(l_b,label='RFC Betting Strategy')
random_line, = plt.plot(l_br,label="Random Betting Strategy")
plt.legend(handles=[rfc_line, random_line])
plt.title("Comparing Betting Strategies")
plt.show()

