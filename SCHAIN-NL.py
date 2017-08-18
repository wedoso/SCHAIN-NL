#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:12:38 2017

@author: yuql216
"""

import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans
import hdbscan
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
# pre-defined parameters
alpha = 1
k=20
n=18346

SA = pd.read_hdf('../pre-calculated_variable/SA.hdf', key='mydata')
SL = pd.read_hdf('../pre-calculated_variable/SL.hdf', key='mydata')
# currently the dimension in each matrix is different, 
# so we are working on the intersection.
user = list(SA.index)
n = len(user)
SL = SL.loc[user]
SL = SL[user]

SA = SA.as_matrix()
SL = SL.as_matrix()

S = alpha*SA + (1-alpha)*SL
#np.save('../pre-calculated_variable/S.npy', S)
print 'Similairty matrix done!'

D = np.sum(S, axis=1)
D = np.diag(D**(-0.5))
#np.save('../pre-calculated_variable/D_minus_half.npy', D)
print 'D matrix done!'

K = np.dot(D,S)
K = np.dot(K,D)
#np.save('../pre-calculated_variable/K.npy',K)
print 'K matrix done!'

#K = np.load('pre-calculated_variable/K.npy')
eigenvals, Z = scipy.linalg.eigh(K, eigvals=(n-k,n-1))
# see http://techqa.info/programming/question/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with-numpy
#np.save('../pre-calculated_variable/Z.npy',Z)
print 'Z matrix done!'

U = np.dot(D, Z)
# normalize U by column
# U_normed = U / U.max(axis=0)
U_normed = (U - U.min(axis=0))/(U.max(axis=0)-U.min(axis=0))
# normalize U by row
U_normed = U_normed / U_normed.max(axis=1).reshape(n,1)
#U_normed = np.transpose(U_normed)
#U_normed = (U_normed - U_normed.min(axis=0))/(U_normed.max(axis=0)-U_normed.min(axis=0))
#U_normed = np.transpose(U_normed)
#np.save('../pre-calculated_variable/U_cut_normed_0809_only_attribute.npy',U_normed)
print 'U_normed matrix done!'


# kmeans for clustering
#kmeans = KMeans(n_clusters=k, random_state=0).fit(U_normed)
#user_index_with_label = kmeans.predict(U_normed)
#Z_optimal = np.zeros((n,k))
#for i in range(n):
#    Z_optimal[i, user_index_with_label[i]] = 1
#np.save('../pre-calculated_variable/Z_optimal.npy', Z_optimal)


# hdbscan* for clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True).fit(U_normed)
# see http://hdbscan.readthedocs.io/en/latest/soft_clustering.html
result = clusterer.labels_
#soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

unique, counts = np.unique(result, return_counts=True)
print np.asarray((unique, counts)).T

tdid_label_df = pd.DataFrame(result, index=user)
tdid_label_df.columns = ['label']
#tdid_label_df.to_csv('../pre-calculated_variable/tdid_label_df_cut_0809_only_attribute.csv')


'''
# for 3d plot
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

fig = pylab.figure()
ax = Axes3D(fig)
#ax.scatter(projection[:,0], projection[:,1], projection[:,2], alpha=0.25)
ax.scatter(projection[:,0], projection[:,1], projection[:,2], s=10, linewidth=0, c=cluster_colors, alpha=0.25)
pyplot.show()
'''



# plot 2s figure
'''
projection = TSNE(n_components=2).fit_transform(U_normed) # has been saved
plt.scatter(*projection.T, s=10, linewidth=0, alpha=0.25)

# plot k-means result
n_color = k
color_palette = sns.color_palette('Paired', n_color)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in user_index_with_label]
plt.scatter(*projection.T, s=10, linewidth=0, c=cluster_colors, alpha=0.25)


# plot HDBSCAN result
_, n_color = np.shape(soft_clusters)
color_palette = sns.color_palette('Paired', n_color)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*projection.T, s=10, linewidth=0, c=cluster_member_colors, alpha=0.25)


# plot HDBSCAN-based soft clustering result
cluster_colors = [color_palette[np.argmax(x)]
                  for x in soft_clusters]
plt.scatter(*projection.T, s=10, linewidth=0, c=cluster_colors, alpha=0.25)


cluster_colors = [sns.desaturate(color_palette[np.argmax(x)], np.max(x))
                  for x in soft_clusters]
plt.scatter(*projection.T, s=10, linewidth=0, c=cluster_colors, alpha=0.25)
'''