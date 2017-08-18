
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy import stats
from geopy.distance import vincenty
from tqdm import tqdm, trange
import datetime
import folium


# In[2]:

# read clustering result
#data_all = pd.read_csv('/home/hadoop/sdl/hdfs_data/tdid_label_df_cut_0809.csv_600')
#data_all.columns = ['tdid', 'label']

data_all = pd.read_csv('/home/hadoop/sdl/hdfs_data/tdid_with_group.csv_849')
data_all.columns = ['label','tdid']

data_only_link = pd.read_csv('/home/hadoop/sdl/hdfs_data/tdid_label_df_cut_0809_only_link.csv_592')
data_only_attribute = pd.read_csv('/home/hadoop/sdl/hdfs_data/tdid_label_df_cut_0809_only_attribute.csv_675')



data_only_attribute.columns = ['tdid', 'label']
data_only_link.columns = ['tdid', 'label']

tdids = list(data_all.tdid.unique())

# read travel info
travel_info = pd.read_csv('/home/hadoop/sdl/hdfs_data/part-00000_856',sep='\t', header=None)
travel_info.columns = ['tdid','date','weekday','hour','lat','lng']
# match travel info with clustering result
travel_info = travel_info[travel_info['tdid'].isin(tdids)]
travel_info.reset_index(drop=True, inplace=True)

# read user attribute info
#tagged_user_data = pd.read_csv('/home/hadoop/sdl/hdfs_data/19498_52506_output_detail_tag_converted_754', sep='\t', header=None)
#tagged_user_data.columns = ['tdid', 'tdid_copy', 'code', 'type', 'rating']
#tagged_user_data = tagged_user_data[['tdid', 'code', 'type']]
#code_col = tagged_user_data['code'].apply(str)
#tagged_user_data['code'] = code_col
#tagged_user_data = tagged_user_data[tagged_user_data['code'].apply(len) == 7]


# In[3]:

# load wifi location
wifi = pd.read_csv('/home/hadoop/sdl/hdfs_data/demo_table.csv_794')


# In[7]:

print len(set(wifi.tdid))
wifi.head()


# In[6]:

#print len(tdids), len(set(travel_info.tdid))
data_match = data_all[data_all['tdid'].isin(list(set(wifi.tdid)))]
def cluster_description(data):
    cluster_data_have = list(data.label.unique())
    cluster_data_have.sort()
    for c in cluster_data_have:
        print c, data[data['label']==c].shape[0]
cluster_description(data_match)


# In[ ]:

# examine cluster major interest
def cluster_major_interest(cluster, data_type):
    if data_type == 'link':
        data = data_only_link
    elif data_type == 'attribute':
        data = data_only_attribute
    else:
        data = data_all
    examine_user = list(data[data['label']==cluster].tdid)
    examine_dataset = tagged_user_data[tagged_user_data['tdid'].isin(examine_user)][['tdid','type']]
    examine_dataset = examine_dataset.groupby('type').count()
    result = examine_dataset.sort_values(['tdid'])
    print result.iloc[-5:]


# In[8]:

def cluster_mean_max_travel_distance(cluster, date, data_type):
    if data_type == 'link':
        tdids = list(data_only_link[data_only_link['label']==cluster].tdid)
    elif data_type == 'attribute':
        tdids = list(data_only_attribute[data_only_attribute['label']==cluster].tdid)
    else:
        tdids = list(data_all[data_all['label']==cluster].tdid)
    cluster_travel = travel_info[travel_info['tdid'].isin(tdids)][['tdid', 'date', 'hour', 'lat', 'lng']]
    cluster_travel = cluster_travel[cluster_travel['date'] ==  date]
    mean_sum_movement, mean_max_movement = np.array([]), np.array([])
   
    for usr in tdids:
        testset = cluster_travel[cluster_travel['tdid'] == usr]
        testset.drop_duplicates(['date','hour'],inplace=True)
        usr_movements = np.array([])
        if len(testset.index) > 1:
            for i in range(len(testset.index)-1):              
                loc_i, loc_j = (testset.iloc[i].lat, testset.iloc[i].lng), (testset.iloc[i+1].lat, testset.iloc[i+1].lng)
                usr_movements = np.append(usr_movements, [vincenty(loc_i, loc_j).miles])
        if len(usr_movements) > 0:
            mean_sum_movement = np.append(mean_sum_movement, usr_movements.sum())
            mean_max_movement = np.append(mean_max_movement, usr_movements.max())
    if len(mean_max_movement)>0:
        return mean_sum_movement.mean(), mean_max_movement.mean()
    else:
        return 0,0
    
    #return mean_max_movement.mean()


# In[9]:

def three_week_cluster_movement(cluster, data_type='all'):
    date_list = ['2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', '2017-06-24', '2017-06-25', 
                '2017-06-26', '2017-06-27', '2017-06-28', '2017-06-29', '2017-06-30', '2017-07-01','2017-07-02',
                 '2017-07-03','2017-07-04','2017-07-05','2017-07-06', '2017-07-07','2017-07-08','2017-07-09']
    m_sum_movement, m_max_movement = np.array([]), np.array([])
    for date in tqdm(date_list):
        s, m = cluster_mean_max_travel_distance(cluster, date, data_type)
        if s:
            m_sum_movement = np.append(m_sum_movement, s)
            m_max_movement = np.append(m_max_movement, m)
    return m_sum_movement, m_max_movement


# In[ ]:

s, m = three_week_cluster_movement(24)
print s, m
print s.mean(), m.mean()


# In[111]:

def frequent_location(cluster, date_range, hour_range, data_type):
    if data_type == 'link':
        tdids = list(data_only_link[data_only_link['label']==cluster].tdid)
    elif data_type == 'attribute':
        tdids = list(data_only_attribute[data_only_attribute['label']==cluster].tdid)
    else:
        tdids = list(data_all[data_all['label']==cluster].tdid)
    cluster_travel = travel_info[travel_info['tdid'].isin(tdids)][['tdid', 'date', 'hour', 'lat', 'lng']]
    cluster_travel = cluster_travel[cluster_travel['date'].isin(date_range)]
    cluster_travel = cluster_travel[cluster_travel['hour'].isin(hour_range)]
    cluster_travel['lat'] = cluster_travel['lat'].round(3)
    cluster_travel['lng'] = cluster_travel['lng'].round(3)
    cluster_travel['location'] = zip(cluster_travel['lat'],cluster_travel['lng'])
    cluster_travel = cluster_travel[['tdid','location']]
    freq_df = cluster_travel.groupby('location').count()
    freq_df.columns = ['count']
    freq_df.sort_values(by='count',ascending=False, inplace=True)
    location_list = list(freq_df.index[:10])
    map_osm = folium.Map(location=[],zoom_start=100)
    i=0
    for loc_info in location_list:
        folium.CircleMarker(location=[loc_info[0], loc_info[1]],
                                        popup=str(freq_df.iloc[i,0]),
                                        radius=10,
                                        color='#3186cc',
                                        fill_color='#3186cc').add_to(map_osm)
        i+=1
    map_osm.save(str(cluster)+'.html')
    print freq_df.iloc[:10]
    return map_osm


# In[112]:

date_range = ['2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', '2017-06-24', '2017-06-25', 
                '2017-06-26', '2017-06-27', '2017-06-28', '2017-06-29', '2017-06-30', '2017-07-01','2017-07-02',
                 '2017-07-03','2017-07-04','2017-07-05','2017-07-06', '2017-07-07','2017-07-08','2017-07-09']
weekend_range = ['2017-06-24','2017-06-25','2017-07-01','2017-07-02','2017-07-08','2017-07-08']
weekday_range = ['2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', 
                '2017-06-26', '2017-06-27', '2017-06-28', '2017-06-29', '2017-06-30', 
                 '2017-07-03','2017-07-04','2017-07-05','2017-07-06', '2017-07-07']
hour_range = range(10,18) 
frequent_location(1, weekday_range, hour_range, 'all')


# In[ ]:




# In[100]:

data_all[data_all['label']==19]['tdid'].to_csv('19_tidids.csv', index=False)


# In[ ]:

def draw_daily_location_map(cluster, date, data_type='all', hour=None):
    if data_type == 'link':
        tdids = list(data_only_link[data_only_link['label']==cluster].tdid)
    elif data_type == 'attribute':
        tdids = list(data_only_attribute[data_only_attribute['label']==cluster].tdid)
    else:
        tdids = list(data_all[data_all['label']==cluster].tdid)
    cluster_travel = travel_info[travel_info['tdid'].isin(tdids)][['tdid', 'date', 'hour', 'lat', 'lng']]
    cluster_travel = cluster_travel[cluster_travel['date'] ==  date]
    
    if hour != None:
        cluster_travel = cluster_travel[cluster_travel['hour'] ==  hour]  
    
    try:
        map_osm = folium.Map(location=[], zoom_start=13)
        for usr in tqdm(tdids):
            testset = cluster_travel[cluster_travel['tdid'] == usr]
            usr_movements = np.array([])
            if len(testset.index) >=1:
                for i in range(len(testset.index)):              
                    loc_i = (testset.iloc[i].lat, testset.iloc[i].lng)
                    usr_movements = np.append(usr_movements, [vincenty(loc_i, loc_j).miles])
                    folium.CircleMarker(location=[testset.iloc[i].lat, testset.iloc[i].lng], 
                                        radius=3,
                                        color='#3186cc',
                                        fill_color='#3186cc').add_to(map_osm)
        map_osm.save(date+'_'+str(cluster)+'.html')
    except:
        print 'no valid gis log'


# In[ ]:

draw_daily_location_map(3, '2017-06-19', hour=0)


# In[ ]:

def draw_hourly_location_map(cluster, data_type='all', hour=None):
    if data_type == 'link':
        tdids = list(data_only_link[data_only_link['label']==cluster].tdid)
    elif data_type == 'attribute':
        tdids = list(data_only_attribute[data_only_attribute['label']==cluster].tdid)
    else:
        tdids = list(data_all[data_all['label']==cluster].tdid)
    date_list = ['2017-06-19', '2017-06-20', '2017-06-21', '2017-06-22', '2017-06-23', '2017-06-24', '2017-06-25', 
                '2017-06-26', '2017-06-27', '2017-06-28', '2017-06-29', '2017-06-30', '2017-07-01','2017-07-02',
                 '2017-07-03','2017-07-04','2017-07-05','2017-07-06', '2017-07-07','2017-07-08','2017-07-09']
    
    cluster_travel = travel_info[travel_info['tdid'].isin(tdids)][['tdid', 'date', 'hour', 'lat', 'lng']]
    cluster_travel = cluster_travel[cluster_travel['date'].isin(date_list)]
    
    if hour != None:
        cluster_travel = cluster_travel[cluster_travel['hour'] ==  hour]  
    
    try:
        map_osm = folium.Map(location=[], zoom_start=13)
        for usr in tqdm(tdids):
            testset = cluster_travel[cluster_travel['tdid'] == usr]
            usr_movements = np.array([])
            if len(testset.index) >=1:
                for i in range(len(testset.index)):              
                    loc_i = (testset.iloc[i].lat, testset.iloc[i].lng)
                    usr_movements = np.append(usr_movements, [vincenty(loc_i, loc_j).miles])
                    folium.CircleMarker(location=[testset.iloc[i].lat, testset.iloc[i].lng], 
                                        radius=3,
                                        color='#3186cc',
                                        fill_color='#3186cc').add_to(map_osm)
        map_osm.save(str(hour)+'_'+str(cluster)+'.html')
    except:
        print 'no valid gis log'



# In[ ]:

draw_hourly_location_map(3, hour=0)


# In[ ]:

def cluster_mean_max_travel_distance_0(cluster, date):
    cluster_travel = travel_info[travel_info['label'] == cluster][['tdid', 'date', 'hour', 'lat', 'lng']]
    cluster_travel = cluster_travel[cluster_travel['date'] ==  date]
    mean_max_movement = np.array([])
    user_lst = list(cluster_travel['tdid'].unique())
    
    for usr in tqdm(user_lst):
        testset = cluster_travel[cluster_travel['tdid'] == usr]
        usr_max_movements = 0
        if len(testset.index) > 1:
            for i in range(len(testset.index)-1):              
                loc_i = (testset.iloc[i].lat, testset.iloc[i].lng)
                loc_j = (testset.iloc[i+1].lat, testset.iloc[i+1].lng)
                movement = vincenty(loc_i, loc_j).miles
                if movement > usr_max_movements:
                    usr_max_movements = movement
        if usr_max_movements >= 0:
            mean_max_movement = np.append(mean_max_movement, usr_max_movements)
    return mean_max_movement.mean()
def clusters_mean_max_travel_distance_0(date, examine_cluster=[19,21,22,24,25]):
    result = str()
    for c in examine_cluster:
        result += 'cluster:'+str(c)+' mean_max_movement on '+str(datetime.date(int(date[:4]),int(date[5:7]),int(date[-2:])).weekday())+':'+str(cluster_mean_max_travel_distance(c, date))
        result += '\n'
    print result

