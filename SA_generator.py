#--coding:utf-8--
import pandas as pd
from pandas import *
import graphlab as gl
import os
import numpy as np
from scipy.spatial import distance


def read_and_modify():
    print("reading file...")
    #os.chdir(r"C:\Users\TEND\Documents\talkingData\community detection\TD_data")
    graph = pd.read_csv("../data/graph_7days_10000", delimiter="\t",
                        names=["bssid", "tdid", "tdid_index", "weight", "lat", "lng"])
    bq = pd.read_csv("../data/biaoqian_dongcheng_tdid_10000", delimiter="\t",
                     names=["tdid", "tdid2", "label_index", "label", "weight"])
    del (bq["tdid2"])
    bq = bq[bq["label_index"] > 1000000]
    return graph, bq


def return_dic(row):
    # combine the label and weight of each row to form a dict
    if long(row["weight"]) > 0:
        return {row["label"]: long(row["weight"])}
    else:
        return {row["label"]: 1L}


def combine_dic(group):
    # combine the dict of each group to a union dict
    dic = {}
    for d in group["dic"].values:
        dic.update(d)
    return Series({"dic": dic})


def tfidf_matrix(tfidf,label_dict):
    # 将所有用户的tfidf转换为一个n*203的矩阵
    m = np.zeros((len(tfidf),len(label_dict)))
    for i in range(len(tfidf)):
        d = tfidf[i]
        for k,v in d.items():
            m[i,label_dict[k]] = v
    return m

def consine_matrix(m,gp_bq):
    # 根据tfidf的矩阵，计算出pair wise的cosine距离
    # SL = pd.DataFrame(0.0,index = gp_bq["tdid"],columns=gp_bq["tdid"])
    # for i in range(len(gp_bq["tdid"])):
    #     print i
    #     num = np.sum(m[i]*m,axis=1)
    #     dem = np.sqrt(np.sum(m[i])*np.sum(m,axis=1))
    #     SL.loc[gp_bq["tdid"][i]] = 1 - num/dem
    # return SL
    cosine_matrix = 1 - distance.cdist(m,m,"cosine")
    SL = pd.DataFrame(cosine_matrix,index=gp_bq["tdid"],columns=gp_bq["tdid"])
    return SL

if __name__ =="__main__":
    group,bq = read_and_modify()

    print "get label dict of each user..."
    bq["dic"] = bq.apply(return_dic,axis=1)

    print "get tfidf for each user..."
    gp_bq_0 = bq.groupby("tdid").apply(combine_dic).reset_index(drop=False)
    gp_bq = gl.SFrame(gp_bq_0)
    gp_bq["tfidf"]=gl.text_analytics.tf_idf(gp_bq["dic"])

    print "calculate the label dict..."
    # 生成label dictionary
    label_dict = {}
    label = bq["label"]
    label = label.drop_duplicates()
    for i,label in enumerate(label.values):
        label_dict[label] = i

    print "calculate similarity matrix..."
    tfidf = tfidf_matrix(gp_bq["tfidf"],label_dict)
    SL = consine_matrix(tfidf,gp_bq)
    SL.to_hdf("../pre-calculated_variable/SA.hdf",key="mydata")