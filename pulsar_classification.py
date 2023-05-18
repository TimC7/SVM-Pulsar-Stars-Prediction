import pandas as pd

dataset = pd.read_csv('pulsar_stars.csv')


size = dataset.shape
print("Dataset size Rows: ", size[0], " Columns: ", size[1])


print("Column names: ", dataset.columns)


print("The Distribution of target_class is: ", dataset['target_class'].value_counts())


print("The percentage distribtuion of target_class is: ", dataset['target_class'].value_counts(normalize=True) * 100)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

cols = dataset.columns[:-1]
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 10.0, gamma = .3, random_state = 0)
classifier.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("This is the accuracy score", accuracy_score(y_test, y_pred))

import pickle
filename = 'SVM_Pulsar_Dataset.sav'
pickle.dump(classifier, open(filename, 'wb'))