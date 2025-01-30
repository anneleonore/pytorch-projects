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
# Step II. Calculate mean vector for each class
##########################################################################################
#Set precision for printing NumPy arrays to 4 decimal places
np.set_printoptions(precision=4)

#Initialize an empty list
mean_vecs = []

#Loop iterates over 1-3
for label in range(1, 4):
    #Calculate the mean of each feature (column) > axis = 0 means calclulate mean along the columns
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')

##########################################################################################
# Step III. Compute within/between-class scatter matrix
##########################################################################################
#Check assumption: class labels are uniformly distributed
print('Class label distribution:', np.bincount(y_train)[1:])

#Important: Assumption is volated so we have to compute the scaled within-class scatter matrix

#A. Within-class scatter matrix (scaled!)
d = 13
S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
    
print('Scaled within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

#B. Between-class scatter matrix
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)  #make column vector

d = 13
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  #make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')

##########################################################################################
# Step IV. Compute eigenvalues and eigenvectors of matrix S_W(-1)S_B
##########################################################################################
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

##########################################################################################
# Step V. Sort eigenvalues in descending order
##########################################################################################
#Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

#Sort the (eigenvalue, eigenvector) tuples from high to low
#sorted(key = ) function returns a new sorted list and uses key to determine the sort order
#lambda k: k[0] > take each individual element (k) in the list and extract the first element k[0]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

##########################################################################################
# Step VI. Create transformation matrix W
##########################################################################################
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

##########################################################################################
# Step VII. Transform input dataset X using matrix W to create new feature subspace
##########################################################################################
#Matrix multiplication between training dataset and matrix W
X_train_lda = X_train_std.dot(w)

#Setting up plotting parameters
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

#Plotting the transformed data
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=f'Class {l}', marker=m)

#Customizing the plot
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

##########################################################################################
# Additional: Plot explained variance
##########################################################################################
#Calculate total variance
tot = sum(eigen_vals.real)

#Calculate the discriminability ratio
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]

#Calculate the cumulative discriminability 
cum_discr = np.cumsum(discr)

#Plot the individual and cumulative discriminability
plt.bar(range(1, 14), discr, align='center', label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative discriminability')
plt.ylabel('Discriminability ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()