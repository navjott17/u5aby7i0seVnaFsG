import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('ACME-HappinessSurvey2020.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=2, criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))


cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print(acc)