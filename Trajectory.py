import numpy as np
import pandas as pd

import TrajectoryDescriptor
import TrajectoryFeatures


class Trajectory:

    def __init__(self, **kwargs):
        self.labels = kwargs.get('labels', ['target'])
        if kwargs['mood'] == 'df':
            self.row_data = kwargs.get('trajectory', pd.DataFrame())
        if kwargs['mood'] == 'csv':
            self.row_data = self.load_data(kwargs)
        self.rows_ = self.row_data.shape[0]
        self.stop_parameters = kwargs.get('stop_parameters', [100, 60, 60, 100])

        self.has_alt = True

        self.duration_features = []
        self.speed_features = []
        self.acc_features = []
        self.jerk_features = []
        self.brate_rate_features = []
        self.brate_features = []
        self.distance_features = []
        self.bearing_features = []
        self.label = ''
        self.noise_no=0
        self.polartheta = []
        self.polarR = []
        self.isPure = False
        self.isInValid = False
        self.filter = True
        # self.labels = []  # "label1:11"
        self.hasAlt = False  # False: we do not have altitude in dataset

        # self.descriptor=TrajectoryDescriptor.TrajectoryDescriptor(trajectory=self.row_data, labels=self.labels)
        #print("smothing..")
        #self.row_data,noise=self.g_hample(self.row_data)
        #self.noise_no=len(noise)
	#if self.noise_no>0 :
	#   print("# noise points:",len(noise))


    def rows(self):
        return self.rows_

    def prediction_actual(self, target):
        self.row_data.loc[:, target + '_prediction'] = self.stat_label()
        return self.row_data.loc[:, [target, target + '_prediction']]

    def get_full_features_column_name(self):
        """
        other = ['isInValid', 'isPure', 'target']

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['speed_'] * len(a2))
        speed_features = map(''.join, zip(a1, a2))

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['distance_'] * len(a2))
        distance_features = map(''.join, zip(a1, a2))

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['acc_'] * len(a2))
        acc_features = map(''.join, zip(a1, a2))

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['bearing_'] * len(a2))
        bearing_features = map(''.join, zip(a1, a2))

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['jerk_'] * len(a2))
        jerk_features = map(''.join, zip(a1, a2))

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['brate_'] * len(a2))
        brate_features = map(''.join, zip(a1, a2))

        a2 = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])
        a1 = np.array(['brate_rate_'] * len(a2))
        brate_rate__features = map(''.join, zip(a1, a2))

        ret = map(''.join, zip(distance_features, speed_features, acc_features, bearing_features,
                               jerk_features, brate_features, brate_rate__features, other))
"""
        return self.descriptor.get_full_features_column_name()

    def trajectory_features(self):

        # other = [self.isInValid, self.isPure, self.stat_label()]

        # return self.distance_features + self.speed_features + self.acc_features + self.bearing_features + self.jerk_features + self.brate_features + self.brate_rate_features + other
        self.descriptor = TrajectoryDescriptor.TrajectoryDescriptor(trajectory=self.row_data, labels=self.labels,
                                                                    stop_parameters=self.stop_parameters)
        ret = self.descriptor.describe()
        return ret

    """
    labels: List of labels for each point
    lat: name of lat column in the dataframe
    lon: name of lon column in the dataframe
    alt: name of alt column in the dataframe
    timeDate: name of time date column in the dataframe
    src: source of the csv file for row_data
    """

    def return_row_data(self):
        return self.row_data

    def load_data(self, **kwargs):
        # lat='lat',lon='lon',alt='alt',timeDate='timeDate',labels=['label1'],src='~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv',seperator=','
        print('loading...')
        lat = kwargs.get('lat', "lat")
        print(lat)
        lon = kwargs.get('lon', "lon")
        print(lon)
        alt = kwargs.get('alt', None)
        print(alt)
        time_date = kwargs.get('timeDate', "timeDate")

        print(time_date)
        labels = kwargs.get('labels', "[label]")
        print(labels)
        src = kwargs.get('src', "~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv")
        print(src)
        separator = kwargs.get('separator', ",")
        print(separator)

        self.labels = labels
        # input data needs lat,lon,alt,timeDate, [Labels]
        self.row_data = pd.read_csv(src, sep=separator, parse_dates=[time_date], index_col=time_date)
        self.row_data.rename(columns={lat: 'lat'}, inplace=True)
        self.row_data.rename(columns={lon: 'lon'}, inplace=True)
        if alt is not None:
            self.row_data.rename(columns={alt: 'alt'}, inplace=True)
        self.row_data.rename(columns={time_date: 'timeDate'}, inplace=True)
        # preprocessing
        # removing NaN in lat and lon

        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lat), :]
        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lon), :]
        for label in labels:
            self.row_data = self.row_data.loc[pd.notnull(self.row_data[label]), :]

        print('Data loaded.')
        return self.row_data

    def load_data_frame(self, data_frame, labels=None):
        if labels is None:
            labels = ['target']
        self.labels = labels
        self.row_data = data_frame
        # preprocessing
        self.pre_processing(labels)

        if (self.row_data.shape[0] < 10):
            return -1
        return 0

    def pre_processing(self, labels):
        # removing NaN in lat and lon
        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lat), :]
        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lon), :]
        for label in labels:
            self.row_data = self.row_data.loc[pd.notnull(self.row_data[label]), :]
        """
        lat_= self.row_data.lat.rolling(3, min_periods=1).median()
        self.row_data.assign(lat=lat_)
        lon_ = self.row_data.lon.rolling(3, min_periods=1).median()
        self.row_data.assign(lot=lon_)

        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lat), :]
        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lon), :]
        """

        return None

    """
    input is a sorted trajectory based on time_date
    output is time difference in second
    """

    """
    start=0,end=15,start_word="<0",end_word=">15",
    dic_speed=['speed[0,0.5]','speed[0.5,1]','speed[1,1.5]','speed[1.5,2]','speed[2.5,4]','speed[4,7]','speed[7,10]','speed[10,13]','speed[13,15]'],
    speed_intr=[(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2.5,4),(4,7),(7,10),(10,13),(13,15)],
    invalid_word="invalid",c=0.6
    """

    def tokenize_sub_trajectory(self, start, end, start_word, end_word, dic_speed, speed_intr, invalid_word, c):
        def in_between(tuple__):
            (tuple_, z) = tuple__
            (x, y) = tuple_
            return True if x <= z <= y else False

        # print(len(speed_intr),len(dic_speed),start_word,c)

        d = list(map(in_between, zip(speed_intr, [c] * len(speed_intr))))
        if np.any(d):
            #   print np.where(d)[0][0]
            a = dic_speed[d.index(True)]
            return a
        else:
            if c < start:
                return start_word
            if c > end:
                return end_word
        print("Ivalid value: ", c, speed_intr)
        return invalid_word

    def sub_trajectory_token(self, sub_traj):
        speed_mean = sub_traj.speed.mean()
        acc_mean = sub_traj.acc.mean()
        bearing_mean = sub_traj.bearing.mean()

        speed_token = self.tokenize_sub_trajectory(start=0, end=15, start_word="s0", end_word="se",
                                                   dic_speed=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'],
                                                   speed_intr=[(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 4),
                                                               (4, 7), (7, 10), (10, 13),
                                                               (13, 15)],
                                                   invalid_word="si", c=speed_mean)

        acc_token = self.tokenize_sub_trajectory(start=-0.15, end=1, start_word="a0", end_word="ae",
                                                 dic_speed=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
                                                 speed_intr=[(-0.15, -0.1), (-0.1, -0.05), (-0.05, -0.025),
                                                             (-0.025, +0.025), (0.025, 0.5), (0.5, 1)],
                                                 invalid_word="ai", c=acc_mean)

        bearing_token = self.tokenize_sub_trajectory(start=0, end=360, start_word="b0", end_word="be",
                                                     dic_speed=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
                                                     speed_intr=[(0, 50), (50, 100), (100, 150), (150, 200), (200, 250),
                                                                 (250, 300),
                                                                 (300, 360)],
                                                     invalid_word="bi", c=bearing_mean)
        subtrajectory_token = speed_token + " " + acc_token + " " + bearing_token + " " + "@@"
        return subtrajectory_token

    def sub_trajectory_speed_token(self, sub_traj):
        speed_mean = sub_traj.speed.mean()

        speed_token = self.tokenize_sub_trajectory(start=0, end=15, start_word="s0", end_word="se",
                                                   dic_speed=['s0.5', 's1', 's1.5', 's2', 's2.5', 's3', 's3.5', 's4',
                                                              's4.5', 's5', 's5.5', 's6', 's6.5', 's7', 's7.5', 's8',
                                                              's8.5', 's9', 's13', 's15', 's30', 's50', 's100'],
                                                   speed_intr=[(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5),
                                                               (2.5, 3), (3, 3.5), (3.5, 4),
                                                               (4, 4.5), (4.5, 5), (5, 5.5), (5.5, 6), (6, 6.5),
                                                               (6.5, 7), (7, 7.5), (7.5, 8), (8, 8.5), (8.5, 9),
                                                               (9, 13),
                                                               (13, 15), (15, 30), (30, 50), (50, 100)],
                                                   invalid_word="si", c=speed_mean)

        subtrajectory_token = speed_token
        return subtrajectory_token

    def sub_trajectory_acc_token(self, sub_traj):
        acc_mean = sub_traj.acc.mean()

        acc_token = self.tokenize_sub_trajectory(start=-0.15, end=1, start_word="N0", end_word="Pe",
                                                 dic_speed=['N-1', 'N-0.5', 'N-0.025', 'NN', 'P0.05', 'P1'],
                                                 speed_intr=[(-0.15, -0.1), (-0.1, -0.05), (-0.05, -0.025),
                                                             (-0.025, +0.025), (0.025, 0.5), (0.5, 1)],
                                                 invalid_word="ai", c=acc_mean)

        subtrajectory_token = acc_token
        return subtrajectory_token

    def sub_trajectory_bearing_token(self, sub_traj):
        bearing_mean = sub_traj.bearing.mean()
        dic_bearing_ = ['b6', 'b12', 'b18', 'b24', 'b30', 'b36', 'b42', 'b48', 'b54', 'b60', 'b66', 'b72', 'b78', 'b84',
                        'b90', 'b96',
                        'b102', 'b108', 'b114', 'b120', 'b126', 'b130', 'b136', 'b142', 'b148', 'b154', 'b160', 'b166',
                        'b172', 'b178', 'b184', 'b190', 'b196', 'b204', 'b210', 'b216', 'b222', 'b228', 'b234', 'b240',
                        'b246', 'b252', 'b258', 'b264', 'b270', 'b276', 'b282', 'b288', 'b294', 'b300', 'b306', 'b312',
                        'b318', 'b324', 'b330', 'b336', 'b342', 'b348', 'b354', 'b360']
        bearing_intr_ = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42), (42, 48), (48, 54),
                         (54, 60), (60, 66),
                         (66, 72), (72, 78), (78, 84), (84, 90), (90, 96), (96, 102), (102, 108), (108, 114),
                         (114, 120),
                         (120, 126), (126, 130), (130, 136), (136, 142), (142, 148), (148, 154), (154, 160), (160, 166),
                         (166, 172), (172, 178), (178, 184), (184, 190), (190, 196), (196, 204), (204, 210), (210, 216),
                         (216, 222), (222, 228), (228, 234), (234, 240), (240, 246), (246, 252), (252, 258), (258, 264),
                         (264, 270), (270, 276), (276, 282), (282, 288), (288, 294), (294, 300), (300, 306), (306, 312),
                         (312, 318), (318, 324), (324, 330), (330, 336), (336, 342), (342, 348), (348, 354), (354, 360)]

        bearing_token = self.tokenize_sub_trajectory(start=0, end=360, start_word="b0", end_word="be",
                                                     dic_speed=dic_bearing_, speed_intr=bearing_intr_,
                                                     invalid_word="bi",
                                                     c=bearing_mean)

        subtrajectory_token = bearing_token
        return subtrajectory_token

    """
    This function discretizes the trajectory by a fix number of records so that the results contain 
    subtrajectories of length n
    """

    def process_sub_traj(self):
        trajectory_text = ""
        d, subtraj = self.discretization_by_records(n=11)
        for i in subtraj:
            trajectory_text = trajectory_text + " " + self.sub_trajectory_token(subtraj[i])
        trajectory_text = trajectory_text + " @"
        return trajectory_text

    def process_speed_sub_traj(self):
        trajectory_text = ""
        d, subtraj = self.discretization_by_records(n=11)
        for i in subtraj:
            trajectory_text = trajectory_text + " " + self.sub_trajectory_speed_token(subtraj[i])
        trajectory_text = trajectory_text + " @"
        return trajectory_text

    def process_acc_sub_traj(self):
        trajectory_text = ""
        d, subtraj = self.discretization_by_records(n=11)
        for i in subtraj:
            trajectory_text = trajectory_text + " " + self.sub_trajectory_acc_token(subtraj[i])
        trajectory_text = trajectory_text + " @"
        return trajectory_text

    def process_bearing_sub_traj(self):
        trajectory_text = ""
        d, subtraj = self.discretization_by_records(n=11)
        for i in subtraj:
            trajectory_text = trajectory_text + " " + self.sub_trajectory_bearing_token(subtraj[i])
        trajectory_text = trajectory_text + " @"
        return trajectory_text

    def discretization_by_records(self, n=11):
        # test again
        idx = np.array(range(self.row_data.shape[0] - 1))
        f = idx[idx % n == 0]
        d = zip(f[:-1], f[1:])
        # test part
        # if f[1:] != self.row_data.shape[0] - 1:
        #    d.append((f[1:][0], self.row_data.shape[0]))
        subtrajectories = {}
        for i in d:
            subtrajectories[i] = self.row_data.iloc[i[0]:i[1], :]
        del f
        del idx
        #
        # subtrajectories[d[1]], d[1]
        return d, subtrajectories

    def point_features(self, smooth_=True, sgn_=False):
        smooth_ = False

        tf = TrajectoryFeatures.TrajectoryFeatures(trajectory=self.row_data, labels=self.labels, smooth=smooth_,
                                                   sgn=sgn_)

        self.row_data = tf.row_data

        return self.row_data

    def toCSV(self, filename):
        self.row_data.to_csv(filename)
        return None

    def to_geojson(self):
        print('todo')
        return None

    def plot(self):
        print('todo')
        return None

    def load_trajectory_from_dataframe(self, df):
        self.row_data = df.copy()

    # 'collected_time','t_user_id','latitude','longitude','altitude'
    def load_trajectory_from_CSV(self, csvFile='~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv'):

        self.row_data = pd.read_csv(csvFile, sep=',', parse_dates=['collected_time'], index_col='collected_time')

    def check_header(self):
        if self.row_data.columns[0] == 'collected_time':
            # assert 'first column is collected time'
            print("d")

    def __del__(self):
        del self.row_data
        # print 'clear memory'

    def g_hample(self,df, k=5, se=3, show_error_bound=False, plot=False, update=True, remove_noise=False):
        lat = df.lat.values
        lon = df.lon.values
        if len(lat) != len(lon):
            raise AssertionError("lat and long must be same length.")

        if len(lat) < (2 * k + 1):
            raise AssertionError("lat must be longer than 2k+1,n=" + str(len(lat)))

        lat_min = np.percentile(lat, 0.05)
        lat_max = np.percentile(lat, 0.95)
        lon_min = np.percentile(lon, 0.05)
        lon_max = np.percentile(lon, 0.95)

        n = len(lat)
        lat_y = lat
        lon_y = lon
        if plot == True:
            plt.figure(figsize=(30, 30))
        # print('n',n)
        lat_s = np.std(lat[0:k])
        lat_m = np.median(lat[0:k])
        lon_s = np.std(lon[0:k])
        lon_m = np.median(lon[0:k])
        noise = []
        for i in range(k):
            if (lat[i] - lat_m < lat_s * se) & (lon[i] - lon_m < lon_s * se):
                continue
            else:
                # print(i)
                noise.append(i)
                if plot == True:
                    plt.scatter(lat[i], lon[i], marker='^', c='r')
                lat_y[i] = lat_m
                if (lat_y[i] > lat_max):
                    lat_y[i] = lat_max
                if lat_y[i] < lat_min:
                    lat_y[i] = lat_min
                lon_y[i] = lon_m
                if (lon_y[i] > lon_max):
                    lon_y[i] = lon_max
                if lon_y[i] < lon_min:
                    lon_y[i] = lon_min
                if plot == True:
                    plt.scatter(lat_y[i], lon_y[i], marker='*', c='r')

        for i in range(n - k):
            ik = i + k
            lat_s = np.std(lat[ik - k:ik + k])
            lat_m = np.median(lat[ik - k:ik + k])
            lon_s = np.std(lon[ik - k:ik + k])
            lon_m = np.median(lon[ik - k:ik + k])
            if plot == True:
                if (show_error_bound == True):
                    e1 = patches.Ellipse((lat[ik], lon[ik]), lat_s * se, lon_s * se, linewidth=2, fill=True, alpha=0.8,
                                         zorder=2)
                    ax = plt.gca()
                    ax.add_patch(e1)
            if (lat[ik] - lat_m < lat_s * se) & (lon[ik] - lon_m < lon_s * se):
                continue
            else:

                noise.append(ik)
                if plot == True:
                    plt.scatter(lat[ik], lon[ik], marker='^', c='r')
                lat_y[ik] = lat_m
                if (lat_y[i] > lat_max):
                    lat_y[i] = lat_max
                if lat_y[i] < lat_min:
                    lat_y[i] = lat_min
                lon_y[ik] = lon_m
                if (lon_y[i] > lon_max):
                    lon_y[i] = lon_max
                if lon_y[i] < lon_min:
                    lon_y[i] = lon_min
                if plot == True:
                    plt.scatter(lat_y[ik], lon_y[ik], marker='*', c='r')

        for i in range(k):
            ik = n - k + i
            if (lat[ik] - lat_m < lat_s * se) & (lon[ik] - lon_m < lon_s * se):
                continue
            else:

                noise.append(ik)
                if plot == True:
                    plt.scatter(lat[ik], lon[ik], marker='^', c='r')
                lat_y[ik] = lat_m
                if (lat_y[ik] > lat_max):
                    lat_y[ik] = lat_max
                if lat_y[ik] < lat_min:
                    lat_y[ik] = lat_min
                lon_y[ik] = lon_m
                if (lon_y[ik] > lon_max):
                    lon_y[ik] = lon_max
                if lon_y[ik] < lon_min:
                    lon_y[ik] = lon_min
                if plot == True:
                    plt.scatter(lat_y[ik], lon_y[ik], marker='*', c='r')

        if plot == True:
            plt.ylim(np.min(lon - 0.0001), np.max(lon + 0.0001))
            plt.xlim(np.min(lat - 0.0001), np.max(lat + 0.0001))
            plt.scatter(lat, lon, c='b', alpha=0.5, s=12)
            plt.scatter(lat_y, lon_y, c='g', alpha=0.5, s=8)
            plt.show()
        if (update == True):
            df = df.assign(lat=lat_y)
            df = df.assign(lon=lon_y)
        if remove_noise == True:
            df = df.drop(df.loc[noise, :].index, axis=0)
        return df, noise
