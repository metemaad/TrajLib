
import pandas as pd
import numpy as np
import TrajectorySegmentation as ts
import Trajectory as tr
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#%matplotlib inline
ts_obj=ts.TrajectorySegmentation()
ts_obj.load_data(lat='latitude',lon='longitude',alt='altitude',time_date='collected_time',
                 labels=['lid'],src='geolife.csv',seperator=',')
print(ts_obj.return_row_data().shape)
segments, trajectorySegments=ts_obj.multi_label_segmentation(labels=['t_user_id','day','transportation_mode'])
print('Number of trajectories in dataset:',len(trajectorySegments))
print('Classes in dataset:',set(ts_obj.return_row_data().transportation_mode))

i=1
features=[]
for seg in range(len(trajectorySegments)):
    if(trajectorySegments[seg].shape[0]>10):

        tr_obj=tr.Trajectory(mood='df',trajectory=trajectorySegments[seg],labels=['transportation_mode'])
        
        tr_obj.point_features() # generate point_features
        f=tr_obj.trajectory_features()# generate trajectory_features
        userid=tr_obj.return_row_data().iloc[0,:].t_user_id

        f.append(userid)
        features.append(np.array(f))
        i=i+1

bearingSet=[ 'bearing_min', 'bearing_max', 'bearing_mean', 'bearing_median', 'bearing_std','bearing_p10','bearing_p25','bearing_p50','bearing_p75','bearing_p90']
speedSet=[ 'speed_min', 'speed_max', 'speed_mean', 'speed_median', 'speed_std','speed_p10','speed_p25','speed_p50','speed_p75','speed_p90']
distanceSet=[ 'distance_min', 'distance_max', 'distance_mean', 'distance_median', 'distance_std','distance_p10','distance_p25','distance_p50','distance_p75','distance_p90']
accelerationSet=['acceleration_min', 'acceleration_max', 'acceleration_mean', 'acceleration_median','acceleration_std','acceleration_p10','acceleration_p25','acceleration_p50','acceleration_p75','acceleration_p90']
jerkSet=[ 'jerk_min', 'jerk_max', 'jerk_mean', 'jerk_median', 'jerk_std','jerk_p10','jerk_p25','jerk_p50','jerk_p75','jerk_p90']
brateSet=[ 'bearing_rate_min', 'bearing_rate_max', 'bearing_rate_mean', 'bearing_rate_median', 'bearing_rate_std','bearing_rate_p10','bearing_rate_p25','bearing_rate_p50','bearing_rate_p75','bearing_rate_p90']
brate_rateSet=[ 'brate_rate_min', 'brate_rate_max', 'brate_rate_mean', 'brate_rate_median', 'brate_rate_std','brate_rate_p10','brate_rate_p25','brate_rate_p50','brate_rate_p75','brate_rate_p90']

targetset=['airplane','bike','boat','bus','car','motorcycle','run','subway','taxi','train','walk']
col=distanceSet+speedSet+accelerationSet+bearingSet+jerkSet+brateSet+brate_rateSet+['isInValid', 'isPure', 'target','userId']

features_set = pd.DataFrame(features,columns=col)
features_set.to_csv('features_cleaned_daily.csv')
   




