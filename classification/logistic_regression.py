import csv
from sklearn.linear_model import LogisticRegression

# read data
def readData(filename, type):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = []
    y = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fs = []
            for f in features:
                if f == 'Sex':
                    fs.append(1.0 if row[f] == 'male' else 0.0)
                else:
                    fs.append(0.0 if row[f] == '' else float(row[f]))
            X.append(fs)
            if type == 'train':
                y.append(0.0 if row['Survived'] == '' else float(row['Survived']))
    return (X, y)

# train
(X_train, y_train) = readData('/Users/puzhang/Desktop/projects/titanic/train.csv', 'train')
model = LogisticRegression()
model.fit(X_train, y_train)
# predict
(X_test, _) = readData('/Users/puzhang/Desktop/projects/titanic/test.csv', 'test')
y_predict = model.predict(X_test)
# read ground truth
y_validate = []
with open('/Users/puzhang/Desktop/projects/titanic/gender_submission.csv') as validateFile:
    reader = csv.DictReader(validateFile)
    for row in reader:
        y_validate.append(float(row['Survived']))
print 'accuracy =', sum([0.0 if y_p != y_v else 1.0 for (y_p, y_v) in zip(y_predict, y_validate)]) / len(y_validate)
