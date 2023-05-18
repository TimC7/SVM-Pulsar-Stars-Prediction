import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

cols = dataset.columns[:-1]
X_test = pd.DataFrame(X_test, columns=[cols])

loaded_svm = pickle.load(open('SVM_Pulsar_Dataset.sav', 'rb'))

y_pred = loaded_svm.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

print("Confusion Matrix: ", cm)

print("Accuracy: ", accuracy_score(y_test, y_pred))

precision = TP / (TP + FP)
print("Precision: ", precision)

recall = TP / (TP + FN)
print("Recall: ", recall)

specificity = TN / (FP + TN)
print("Specificity: ", specificity)