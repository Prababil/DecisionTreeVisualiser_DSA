import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree

def get_decision_tree(df:pd.DataFrame):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    rf_clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 666, min_samples_split=20, max_features=1)
    rf_clf.fit(X_train, y_train)
    predictions = rf_clf.predict(X_test)
    print(f'The accuracy score for the given model is {accuracy_score(y_test, predictions)*100:.2f}%')
    plot_tree(rf_clf.estimators_[0],
          filled=True, impurity=True, 
          rounded=True)
    plt.title("One of the trees from the Random Forest Method")
    plt.savefig('images/Tree.png')
