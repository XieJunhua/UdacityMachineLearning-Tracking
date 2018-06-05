# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('census.csv')

# display(data.head(n=1))
income_raw = data['income']
income = income_raw.apply(lambda x: 0 if str(x) == '<=50K' else 1)
features_raw = data.drop('income', axis = 1)
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
features = pd.get_dummies(features_raw)
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.33, random_state=0, stratify=income)

# clf = XGBClassifier().fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred=y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
clf = XGBClassifier()
learning_rate = [0.1, 0.2, 0.3]
param_grid = {
  "learning_rate":learning_rate
  # "max_depth": [6, 7, 8, 9, 10],
  # "min_child_weight": range(1, 10)

}
# param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(clf, param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

y_pred = grid_result.predict(X_test)
accuracy = accuracy_score(y_test, y_pred=y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



