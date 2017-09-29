
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

comb = pd.concat[train, test]

comb.loc[comb["Sex"] == "female", "Sex"] = 1
comb.loc[comb["Sex"] == "male", "Sex"] = 0
comb.loc[comb["Age"].isnull(), "Age"] = np.floor(comb["Age"].mean())

comb["Age"] = comb["Age"].astype(int)
comb["AgeInt"] = pd.cut(comb["Age"], 5)

def getInterval(age):
	for i in range(0,10):
		if age <=16 * (i + 1):
			return i

comb["AgeInt"] = comb["Age"].apply(lambda x: getInterval(x))



def getTitle(title):
	return title.split(",")[1].split(" ")[1]

comb["Title"] = comb["Name"].apply(lambda x: getTitle(x))


def setTitle(argument):
	switcher = {
		"Mr.": "Mr",
		"Miss.": "Miss",
		"Mrs.": "Mrs",
		"Master.": "Master",
		"Mme.": "Miss",
		"Ms.": "Miss",
		"Mlle.": "Miss",
	}
	return switcher.get(argument, "Others")
comb["Title"] = comb["Title"].apply(lambda x: setTitle(x))


comb = comb.drop(["Name"], axis = 1)


def getTitleId(title):
	if title == "Mr":
		return 0
	if title == "Miss":
		return 1
	if title == "Mrs":
		return 2
	if title == "Master":
		return 3
	return 4
comb["Title"] = comb["Title"].apply(lambda x: getTitleId(x))

comb.loc[comb["Embarked"].isnull(), "Embarked"] = comb["Embarked"].value_counts().argmax()

def getEmbarkedId(embarked):
	if embarked == "C":
		return 0
	if embarked == "Q":
		return 1
	if embarked == "S":
		return 2

comb["Embarked"] = comb["Embarked"].apply(lambda x: getEmbarkedId(x))

comb["Fare"] = comb["Fare"].astype(int)
comb["FareInt"] = pd.qcut(comb["Fare"], 4)

def getFareId(fare):
	if fare <= 7.91:
		return 0
	if fare <= 14.454:
		return 1
	if fare <= 31:
		return 2
	return 3

comb["FareInt"] = comb["Fare"].apply(lambda x: getFareId(x))
comb = comb.drop("Fare", axis = 1)

comb = comb.drop("Ticket", axis = 1)

comb = comb.drop("PassengerId", axis = 1)

X = train.drop("Survived", axis = 1)
y = train["Survived"]


from sklearn import linear_model

logistic = linear_model.LogisticRegression()
logistic.fit(X, y)
score = logistic.score(X, y) * 100
score

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

log_cv = linear_model.LogisticRegression()
log_cv.fit(X_train, y_train)
score = log_cv.score(X_test, y_test) * 100
score


from sklearn import svm

SVM = svm.SVC()
SVM.fit(X, y)
score = SVM.score(X, y) * 100
score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

SVM_cv = svm.SVC()
SVM_cv.fit(X_train, y_train)
score = SVM_cv.score(X_test, y_test) * 100
score


from sklearn import tree

dt = tree.DecisionTreeClassifier()
dt.fit(X, y)
score = dt.score(X, y) * 100
score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

dt_cv = tree.DecisionTreeClassifier()
dt_cv.fit(X_train, y_train)
score = dt_cv.score(X_test, y_test) * 100
score
