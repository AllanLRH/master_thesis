# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:17:35 2016

@author: edin
"""

import multiprocessing as mp
import numpy as np
import pandas as pd
import random    
import datetime

from sklearn import cluster
from collections import Counter

from math import radians, cos, sin, asin, sqrt, degrees, atan2

longitude2meters = 111319
latitude2meters = 110946
latlonscalefactor = (111338/62738.)
    


def my_haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000




def get_random_user():
    """Returns a random user """
    user_list = get_userlist()
    n = len(user_list)-1
    return user_list[random.randint(0,n)]

def get_userlist():
    """Returns a list of all the users in the SF project"""
    
    with open('userlist.txt','r') as f:
        users = f.readlines()
    user_list = [user[:-1] for user in users]
    return user_list
    
def get_data(user,datatype='gps_log', version = 'normal'):
    """Gets the data for a given user.

    The multiprocessing module is used to speed up the conversion of strings 
    containing the data to dictionaries. 
    """

    #Open file containing the data
    if version == 'normal':
        with open('/lscr_paper/amoellga/Data/Telefon/userfiles/%s/%s.txt'%(user,datatype), 'r') as f:
            data = list(f)
    elif version == 'Spyder':
        with open('/run/user/1000/gvfs/sftp:host=paper.biocmplx.nbi.dk,user=edin/lscr_paper/amoellga/Data/Telefon/userfiles/%s/%s.txt'%(user,datatype), 'r') as f:
            data = list(f)

    #Convert from str to list
    pool = mp.Pool(processes = 40)
    data_dict = pool.map(eval, [data_point for data_point in data])
    pool.close()
    pool.join()
    
    return data_dict

# def get_data(user,datatype='gps_log', version = 'normal'):
#     """
#     Gets the data for a given user. I removed the use of multiprocessing since I need it at a higher level.
#     """

#     #Open file containing the data
#     if version == 'normal':
#         with open('/lscr_paper/amoellga/Data/Telefon/userfiles/%s/%s.txt'%(user,datatype), 'r') as f:
#             data = list(f)
#     elif version == 'Spyder':
#         with open('/run/user/1000/gvfs/sftp:host=paper.biocmplx.nbi.dk,user=edin/lscr_paper/amoellga/Data/Telefon/userfiles/%s/%s.txt'%(user,datatype), 'r') as f:
#             data = list(f)

#     data_dict = []
#     for datapoint in data:
#         data_dict.append(eval(datapoint))   
#     return data_dict

def purge_data(data, min_time = 1104537600, min_accuracy = 50 ):
    """Removes all results that have invalid timestamps and low accuracy.
    
    Invalid timestamp means anything before 2005.         
    """
    purged_data = []
    for data_point in data:                          
        if data_point['accuracy'] < min_accuracy:
            if data_point['timestamp'] > min_time:      
                purged_data.append(data_point)
    return purged_data

def add_slight_offset(data, length):
    lat_grid_size = length/float(latitude2meters)
    lon_grid_size = length/float(longitude2meters)*latlonscalefactor

    lat_offset = np.random.uniform(0,lat_grid_size)
    lon_offset = np.random.uniform(0,lon_grid_size)
    
    offset_data = []
    for data_point in data:
        data_point['lat'] += lat_offset
        data_point['lon'] += lon_offset                          
        offset_data.append(data_point)
    return offset_data

def get_grid(data, length = 100, units='degrees'):
    """Transforms the GPS data to locations on a square grid
    
    Each element in grid contains the x,y coordinates for the square grid location
    corresponding to a datapoint. The length parameter sets the spatial scale.     
    """
    grid_y = []   
    grid_x = []
    timestamps = [] 


    lat_grid_size = length/float(latitude2meters)
    lon_grid_size = length/float(longitude2meters)*latlonscalefactor

    if units == 'degrees':
        for data_point in data:
            grid_y.append( int( data_point['lat']  // lat_grid_size ))
            grid_x.append( int( data_point['lon'] // lon_grid_size ))
            timestamps.append(data_point['timestamp'])
    elif units == 'meters':
        for data_point in data:
            grid_y.append( int( (data_point['lat']/latitude2meters)  // lat_grid_size ))
            grid_x.append( int( (data_point['lon']/longitude2meters)  // lon_grid_size ))
            timestamps.append(data_point['timestamp'])

    grid = np.column_stack( ( np.array( grid_y ), np.array( grid_x ) ) )
    return grid, np.array(timestamps)

    
def get_locations(grid):
    """Converts the grid locations to a list of locations """    

    locations = []
    # This dictionary holds all the locations that the user has already visited
    dic_loc = {}
    # The variable will hold the number used to label the next location
    new_loc = 1
    # Converts the grid matrix to a list
    grid = grid.tolist()
    
    for yx in grid:
        if str(yx) in dic_loc:  
            # If the yx coordinates have been encountered before then append the
            # number used to identify that square grid to the locations list
            locations.append( dic_loc[str(yx)] )
        else:
            # Otherwise label this yx coordinate with the next number
            dic_loc[ str(yx) ] = new_loc
            locations.append( new_loc )
            new_loc += 1
            
    return np.array(locations)


def get_Dataframe(locations, places, timestamps, grid=False):
    timestamps_d = [datetime.datetime.fromtimestamp(i) for i in timestamps]    

    if grid:
        return pd.DataFrame(locations, index = timestamps_d)    
    else:
        p = np.swapaxes(places,0,1)
        a = pd.DataFrame(locations, index = timestamps_d)    
        b = pd.DataFrame(p[0], index = timestamps_d)
        c = pd.DataFrame(p[1], index = timestamps_d)        
        return pd.concat([a,b,c], axis=1) 

def custom_resampler(array_like):
    if len(array_like) > 0:
        c = Counter(array_like)       #Count everything in the locations_holder
        m = c.most_common()[0][1]           #Number of measurments at the most frequent location
        r = [k[0] for k in c.most_common() if k[1] == m]    #All equifrequent locations
        return random.choice(r)
    else:
        return -1

def get_binned_timeseries(dataframe, dt = 60):
    binned_timeseries = dataframe.resample(str(dt)+'T', how=custom_resampler)
    return binned_timeseries
    
def get_locations_history(binned_ts):
    return binned_ts.values.astype(int)

def remove_duplicates_from_timeseries(df):
    index_of_same = df == df.shift(1)
    index_of_same = index_of_same[index_of_same == True]
    return df.drop(index_of_same.index)


def remove_repeat_duplicates_from_trace(df):
    index_of_same = df == df.shift(1)
    index_of_same = index_of_same[index_of_same == True]
    return df.drop(index_of_same.dropna().index)


##############################################################################################
# Rewriting everything using pandas for speed 
##############################################################################################

def get_data_with_pd(user,datatype='gps_log'):
    filename = '/lscr_paper/amoellga/Data/Telefon/userfiles/%s/%s.txt'%(user,datatype)
    data = pd.read_csv(filename, header=None, sep="[:,]", engine='python')
    if data.empty:
        return 'No data for this user'
    else:
        data[13] = data[13].map(lambda x: float(x[:-1]))
        return data.drop([0, 2, 4,5,6, 7,8,10,11,12], axis=1)

def purge_data_with_pd(data):
    data = data[data[13] <= 50]
    data = data[data[1] > 1325376000]
    data = data.drop([13], axis=1)
    return data

def get_grid_with_pd(data, length=100):
    data[9] = data[9].map(lambda x: int(x*latitude2meters) // length) 
    data[3] = data[3].map(lambda x: int(x*longitude2meters) // length)
    return data.loc[:, 3:9].as_matrix()

def get_df_using_pd(locations, data):
    timestamps_d = data[1].map(lambda x:datetime.datetime.fromtimestamp(x))    
    return pd.Series(locations, index = timestamps_d)

def get_DataFrame_using_pd(user, datatype='gps_log', length=100):
    data = get_data_with_pd(user)
    if type(data) == str:
        return 'No data for this user'
    else: 
        data = purge_data_with_pd(data)
        grid = get_grid_with_pd(data, length=100)
        locations = get_locations(grid)
        df = get_df_using_pd(locations, data)
        return df


##############################################################################################
# Stationary places
##############################################################################################

def get_places(results,window=15, POI=True):
    timestamps = []
    places = []
    
    if POI:
        window = window * 60 #Convert window from minutes to seconds

        upper = int(window*0.1)
        lower = -int(window*0.1)

        place_cut = int(window*0.5)
        
        t0 = results[0]['timestamp']
        lon0 = results[0]['lon']
        lat0 = results[0]['lat']
        t1 = results[1]['timestamp']
        lon1 = results[1]['lon']
        lat1 = results[1]['lat']
        dt1 = t1 - t0 - window
        Dt1 = abs( dt1 )
        for result in results[2:]:
            t2 = result['timestamp']
            lon2 = result['lon']
            lat2 = result['lat']
            dt2 = t2 - t0 - window
            Dt2 = abs( dt2 )
            if Dt2 > Dt1:
                if dt1 > lower:
                    if dt1 < upper:
                        dr = my_haversine(lon0, lat0, lon1, lat1)
                        if dr < place_cut:
                            timestamps.append( ( t0 + t1 ) / 2. )
                            places.append(((lon0+lon1)/2.,(lat0+lat1)/2.))   
                t0 = t1
                lon0 = lon1
                lat0 = lat1
            t1 = t2
            lon1 = lon2
            lat1 = lat2
            dt1 = t1 - t0 - window
            Dt1 = abs( dt1 )
    else:
        for result in results:
            time = result['timestamp']
            lon = result['lon']
            lat = result['lat']
            timestamps.append( time )
            places.append(((lon0+lon1)/2.,(lat0+lat1)/2.))   
        
    return np.array(timestamps),np.array(places)


def cluster_places(places,eps=50,min_samples=4,view=False):
 
    kms_per_radian = 6371.0088
    eps_in_meters = eps
    eps = eps_in_meters*0.001 / kms_per_radian

    db = cluster.DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(places))
    labels = db.labels_
    
    if view:
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
            class_member_mask = (labels == k)
            xy = places[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=5)
            xy = places[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=6)
        plt.show()
    
    return labels
    
