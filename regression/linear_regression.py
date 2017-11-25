import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt, pow

# read data
def readData(filename):
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    X = []
    y = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fs = []
            for f in features:
                fs.append(0.0 if row[f] == '' else float(row[f]))
            X.append(fs)
            y.append(0.0 if row['price'] == '' else float(row['price']))
    return (X, y)

(X, y) = readData('/Users/puzhang/Desktop/projects/house_price/kc_house_data.csv')
train_size = int(0.9 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# train
model = LinearRegression()
model.fit(X_train, y_train)
# predict
y_predict = model.predict(X_test)

# calculate the rmse
rmse = 0.0
for i in range(len(y_predict)):
    rmse += pow((y_predict[i] - y_test[i]) / y_test[i], 2)
print 'rmse =', sqrt(rmse / len(y_predict))
