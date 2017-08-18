#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:46:45 2017
For link-based similarity matrix
@author: yuql216
"""
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import trange
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance


if __name__ == '__main__':
    embedding_d = 20
    # load graph file
    graph_7days_10000 = pd.DataFrame.from_csv('../data/graph_7days_10000', 
                                              sep='\t', header=None)
    graph_7days_10000.columns = ['tdid','tdid_idex','weight','lat','long']
    graph = graph_7days_10000[['tdid','weight']]
    
    weighted_edge_list = [(index, row[0], row[1]) for index, row in graph.iterrows()]

    # generate a weighted graph
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edge_list)
    
    labeled_user = pd.DataFrame.from_csv('../data/19498_52506_output_detail_tag_converted', sep='\t', header=None)
    labeled_user.columns = ['tdid', 'code', 'catagory', 'weight']
    user = labeled_user.index.unique()

    try: # check whether SL file exists
        SL = pd.read_hdf('../pre-calculated_variable/SL.hdf', key='mydata')
        
        # do some operations
        
        X=np.load('../pre-calculated_variable/embedding20.npy')
        X = pd.DataFrame(X, index=G.nodes(), columns=range(20))
        X = X.loc[user]
        kmeans = KMeans(n_clusters=100, random_state=0).fit(X)
        labeled_result = kmeans.predict(X.iloc[range(18454)])
#        # label:0
#        plt.plot(range(18454), SL.iloc[0], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[1], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[2], alpha=0.4)
#        # label: 3
#        plt.plot(range(18454), SL.iloc[132], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[191], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[209], alpha=0.4)
#        # label: 4
#        plt.plot(range(18454), SL.iloc[64], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[367], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[558], alpha=0.4)
#        # label: 5
#        plt.plot(range(18454), SL.iloc[36], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[43], alpha=0.4)
#        plt.plot(range(18454), SL.iloc[334], alpha=0.4)
            

        
    except:    
        try: 
            result = np.load('../pre-calculated_variable/embedding20.npy')
            print 'embedding file found!'
        except:
            print 'embedding file not found; start manifold learning...'
            A = nx.adjacency_matrix(G, nodelist=G.nodes())
            result = manifold.spectral_embedding(A, n_components=embedding_d)
            # see http://blog.csdn.net/robberm/article/details/9032949
            np.save('../pre-calculated_variable/embedding20.npy', result)
            print 'manifold learning completed! File saved.'
        
        #result = np.load('./pre-calculated_variable/embeding.npy')
#        # E distance
#        SL = pd.DataFrame(0.0, index=G.nodes(), columns=G.nodes())
#        for i in trange(len(G.nodes())):
#            SL.loc[G.nodes()[i]] = np.sqrt(np.sum((result-result[i])**2, axis=1))
        
        # cosine distance matrix
        cosine_matrix = distance.cdist(result, result, 'cosine')
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        SL = pd.DataFrame(cosine_matrix, index=G.nodes(), columns=G.nodes())
        
        # filter out wifi nodes
        SL = SL.loc[user]
        SL = SL[user]      
        
        
        SL = 1 - SL # cosine similarity
        # SL = 0.5 + 0.5*SL
        SL[SL<0]=0
        #SL[SL>1]=1
        # rescale SL matrix
        #SL = (SL-SL.min())/(SL.max()-SL.min())
        SL.to_hdf('../pre-calculated_variable/SL.hdf', key='mydata')


        #cosine_matrix0 = 0.5+0.5*(1-distance.cdist(result, result, 'cosine'))


