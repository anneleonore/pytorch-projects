#Linear Discriminant Analysis in scikit-learn

##########################################################################################
# 1. Import packages
##########################################################################################
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from matplotlib.colors import ListedColormap
#from python_environment_check import check_packages
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

print("Success")

##########################################################################################
# 2. General settings
##########################################################################################
#Download data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
#Set column names
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
#Print sample of data
print(df_wine.head())

#Split data into features (X) and target (y)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

#Split data into training (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

##########################################################################################
# Step I. Standardization
##########################################################################################
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train) #fit_transform estimates parameters and applies transformation
X_test_std = sc.transform(X_test) #transform only applies transformation

##########################################################################################
# Step II. Dimensionality reduction using LDA
##########################################################################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Create LDA object with two features
lda = LDA(n_components=2)

#Fit LDA model to the training and test data
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

##########################################################################################
# Step III. Training a logistic regression model
##########################################################################################
from sklearn.linear_model import LogisticRegression

#Create logistic regression object
lr = OneVsRestClassifier(LogisticRegression(random_state=1, solver='lbfgs'))

#Train logistic regression model using transformed training data (X_train_lda)
lr = lr.fit(X_train_lda, y_train)

##########################################################################################
# Step IV. Plot decision regions
##########################################################################################
from matplotlib.colors import ListedColormap

#Define plot_decision_regions function
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #Setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #Plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

#Apply plot_decision_regions to training data
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#Apply plot_decision_regions to test data
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
