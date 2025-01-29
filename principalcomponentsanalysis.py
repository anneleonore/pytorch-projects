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
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)

##########################################################################################
# Step I. Standardization
##########################################################################################
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train) #fit_transform estimates parameters and applies transformation
X_test_std = sc.transform(X_test) #transform only applies transformation

##########################################################################################
# Step II. Construct covariance matrix
##########################################################################################
cov_mat = np.cov(X_train_std.T) #transpose because we need the features as rows (not columns)

##########################################################################################
# Step III. Decompose covariance matrix into eigenvalues and eigenvectors
##########################################################################################
#linal.eig performs the eigencomposition 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', eigen_vals)

##########################################################################################
# Step IV. Sort eigenvalues by decreasing order to rank eigenvectors
##########################################################################################
#Make a list of (eigenvalue, eigenvector) tuples (list comprehension)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

#Sort the (eigenvalue, eigenvector) tuples from high to low
#lambda arguments: expression > k refers to each tuple (eigenvalue, eigenvector) in eigen_pairs
eigen_pairs.sort(key=lambda k: k[0], reverse=True) #sorting applies to all eigenvalues

##########################################################################################
# Step V. Select k eigenvectors (k largest eigenvalues) 
##########################################################################################
print("In this example, we select the first two eigenvectors.")

##########################################################################################
# Step VI. Construct projection matrix W from top k eigenvectors
##########################################################################################
#We access the eigenvectors using eigen_pairs[0][1] and eigen_pairs[1][1]
#We are adding a newaxis to each eigenvector to make sure they are 2D arrays
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

##########################################################################################
# Step VII. Transform input data X using projection matrix W to obtain data in new feature subspace
##########################################################################################
#Multiply x_train_std (standardized training dataset) with projection matrix W 
X_train_pca = X_train_std.dot(w)

##########################################################################################
# Step VIII. Visualize the transformed dataset through a scatterplot
##########################################################################################
#Set colors and markers
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

#Check how many unique classes we have in y_train (three)
print("Unique classes in y_train:", np.unique(y_train))

#l represents unique class labels from y_train
for l, c, m in zip(np.unique(y_train), colors, markers): #zip to iterate over many lists
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=f'Class {l}', marker=m)

#Set scatterplot xlabel, ylabel, legend and show result
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

##########################################################################################
# Extra visual: Plotting explained variance ratios
##########################################################################################
#I. Calculate total variance
tot = sum(eigen_vals)

#II. Calculate explained variance ratio
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

#III. Calculate cumulative explained variance
cum_var_exp = np.cumsum(var_exp)

#IV. Step plot: Plotting the explained variance ratios
plt.bar(range(1, 14), var_exp, align='center', label='Individual explained variance')

#V. Step plot: Plotting the cumulative explained variance
plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')

#Customizing the plot
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()