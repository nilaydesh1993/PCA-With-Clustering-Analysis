"""
Created on Fri Apr 24 11:41:07 2020
@author: DESHMUKH
DIMENSION REDUCTION (PCA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

pd.set_option('display.max_columns',50)

# ===================================================================================
# Business Problem :- Perform Principal component analysis and perform clustering.
# ===================================================================================

wine = pd.read_csv("wine.csv")
wine.info()
wine.head()
wine = wine.drop('Type',axis = 1)
wine.head()
wine.isnull().sum()

# Summary
wine.describe()

# Boxplot
wine.boxplot(notch='True',patch_artist=True,grid=False);plt.xticks(fontsize=6,rotation=90)

############################### -  Principle Component Analysis - ###############################

# Standardized data for PCA (mandatory)
wine_std = scale(wine)
wine_std

# Appling PCA
pca = PCA(n_components = 13)
wine_pca = pca.fit_transform(wine_std)

# Selection Number of PCA
## Variations
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

## Variance plot for PCA components obtained 
plt.plot(var1,'o-',color="purple")

# Scatter Plot
plt.scatter(wine_std[:,0],wine_std[:,1])

# As help of Variance selecting 3 PCA.
wine_final = wine_pca[:,0:3] 

############################### -  Hierarchical Clustering - ###############################

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Creating linkage for Dendogrma
L = sch.linkage(wine_final, method = "complete", metric = "euclidean")

# Dendogram
plt.figure(figsize=(20, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(L ,leaf_rotation=90, leaf_font_size=5)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = "complete" , affinity = "euclidean").fit(wine_final)

# Converting result into Dataframe or Series
wine_hie_labels = pd.DataFrame((h_complete.labels_),columns = ['cluster'])

# Concating lable dataframe into original data frame
wine_hie_final = pd.concat([wine_hie_labels,wine],axis=1)

# Getting aggregate mean of each cluster
wine_hie_final.iloc[:,1:].groupby(wine_hie_final.cluster).mean()

############################### -  K-means Clustering - ###############################

from sklearn.cluster import KMeans

# Elbow curve 
sse = []
k_rng = range(2,15)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(wine_final)
    sse.append(km.inertia_)

# Scree plot or Elbow Curve
plt.plot(k_rng,sse,'H--',color = 'G');plt.ylabel('Sum of squared error');plt.xlabel('K')

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(wine_final)

# Converting numpy array into pandas Dataframe as labels
md=pd.DataFrame((model.labels_),columns = ['cluster']) 

# Concating lable dataframe into original data frame
wine_kmean = pd.concat([md,wine],axis=1)
wine_kmean.head()

# Getting aggregate mean of each cluster
wine_kmean.iloc[:,1:].groupby(wine_kmean.cluster).mean()

# Scatter plot
x = wine_final[:,0]
y = wine_final[:,1]
plt.scatter(x, y, c=wine_kmean.cluster, cmap = 'gnuplot')

# Scatter plot type 2
winess = pd.DataFrame(wine_final)
winess.plot(0,1,c=wine_kmean.cluster, kind="scatter",cmap='brg')


                    # ---------------------------------------------------- #

