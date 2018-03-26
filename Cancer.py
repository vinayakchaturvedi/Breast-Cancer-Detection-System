#%%
# data preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets  using pandas library
dataset = pd.read_csv('data1.csv')    
X_train = dataset.iloc[: , 1:10].values
y_train = dataset.iloc[: , len(dataset.columns)-1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 5:6])
X_train[:, 5:6] = imputer.transform(X_train[:, 5:6])

"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#%%
# Run only one algorithm at a time

#%%

# Fitting SVM to the Training set                          # Accuracy = 96.865%
from sklearn.svm import SVC
classifier = SVC(C = 1, kernel = 'rbf', random_state = 0, gamma = 0.1)         
classifier.fit(X_train, y_train)


#%%

# Fitting K-NN to the Training set                          # Accuracy = 96.724%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#%%

# Fitting Random Forest Classifier to the Training set      # Accuracy = 96.722%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#%%

# Fitting Logistic Regression to the Training set           # Accuracy = 96.722%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#%%

# Fitting Naive Bayes to the Training set                  # Accuracy = 95.860%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#%%

# Fitting Decision tree Classifier to the Training set     # Accuracy = 93.438%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#%%

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

#%%

# Applying Grid Search to find the best model and the best parameters    #improvements
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
#%%