#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:14:06 2017

@author: yuql216
"""

import pandas as pd
data_use_all = pd.read_csv('../data/tdid_with_group.csv')
data_use_attribute = pd.read_csv('../data/only_attribute.csv')
data_use_link = pd.read_csv('../data/only_link.csv')

data_use_all.columns = ['label', 'tdid']
data_use_attribute.columns = ['tdid', 'label']
data_use_link.columns = ['tdid', 'label']

def cluster_description(data):
    cluster_data_have = list(data.label.unique())
    cluster_data_have.sort()
    for c in cluster_data_have:
        print c, data[data['label']==c].shape[0]

def comparison(cluster_to_examine, data):
    #no_link_cluster = [0,1,2,3,4,5,9,11]
    cluster_data_have = list(data.label.unique())
    cluster_data_have.sort()
    print 'cluster have:', cluster_to_examine.shape[0]
    for c in cluster_data_have:
        user_in_cluster_c = data[data['label']==c]
        over_lap_nuser = (user_in_cluster_c['tdid'].isin(cluster_to_examine.tdid)).sum()
        if over_lap_nuser: 
            print c, over_lap_nuser, data[data['label']==c].shape[0]
