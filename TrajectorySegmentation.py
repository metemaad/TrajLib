import numpy as np
import pandas as pd


# pd.options.mode.chained_assignment = 'raise'
# pd.options.mode.chained_assignment = None


class TrajectorySegmentation:
    """
    note that row_data input should be indexed by the timeDate of trajectory
    meanTime:
        is the indicator for split sequence of data to sub trajectories
        meanTime default is the mean of duration of the input trajectory
        
    minimumNumberOfitemsInTraj:
        is the minimum number of records allowed to make a subtrajectory
    """

    def __init__(self, rowData=pd.DataFrame()):
        self.row_data = rowData

    def return_row_data(self):
        return self.row_data

    def load_dataframe(self, dataframe):
        self.row_data = dataframe

    def load_data(self, **kwargs):
        # lat='lat',lon='lon',alt='alt',time_date='time_date',labels=['label1'],src='~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv',seperator=','
        print('loading...')
        lat = kwargs.get('lat', "lat")
        print(lat)
        lon = kwargs.get('lon', "lon")
        print(lon)
        alt = kwargs.get('alt', None)
        print(alt)
        time_date = kwargs.get('time_date', "time_date")

        print(time_date)
        labels = kwargs.get('labels', "['target']")
        print(labels)
        src = kwargs.get('src', "~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv")
        print(src)
        seperator = kwargs.get('seperator', ",")
        print(seperator)

        self.labels = labels
        # input data needs lat,lon,alt,time_date, [Labels]
        # ,nrows=80000
        #self.row_data = self.row_data.drop_duplicates(time_date)
        self.row_data = pd.read_csv(src, sep=seperator, parse_dates=[time_date],index_col=time_date)
        #self.row_data = self.row_data.drop_duplicates(['t_user_id',time_date])
        #self.row_data.set_index(time_date)

        self.row_data.rename(columns={(lat): ('lat')}, inplace=True)
        self.row_data.rename(columns={(lon): ('lon')}, inplace=True)
        if alt != None:
            self.row_data.rename(columns={(alt): ('alt')}, inplace=True)
            self.hasAlt = True
        self.row_data.rename(columns={(time_date): ('time_date')}, inplace=True)
        #self.row_data = self.row_data.drop_duplicates(['t_user_id','time_date'])
        #self.row_data = self.row_data.set_index('time_date')

        # sort data first
        #self.row_data=self.row_data.sort_index()
        self.row_data['day'] = self.row_data.index.date

        # preprocessing
        # removing NaN in lat and lon

        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lat), :]

        self.row_data = self.row_data.loc[pd.notnull(self.row_data.lon), :]

        for label in labels:
            self.row_data = self.row_data.loc[pd.notnull(self.row_data[label]), :]

        print('Data loaded.')
        return self.row_data

    def multi_label_segmentation(self, labels=['t_user_id', 'transportation_mode'],max_points=1000,max_length=False):

        segments = []
        start = 0
        end = self.row_data.shape[0] - 1

        print(self.row_data.shape)

        segments.append([start, end])

        for label in labels:
            new_segments = []
            for seg in range(len(segments)):
                start = segments[seg][0]
                end = segments[seg][1]

                stseg = self.row_data.iloc[start:end, :]
                s, sta = self.onelabelsegmentation(stseg, label)
                j=0
                sn=[]
                for x in range(len(s)):
                    #discritize if more than max_points=1000
                    #print(s[x][0] + start, s[x][1]+start,start)
                    leng = s[x][1] - s[x][0]
                    start2 = start + s[x][0]
                    if (max_length==False)&(leng >=max_points):
                        leng=s[x][1]-s[x][0]

                        idx = np.array(range(leng ))
                        f = idx[idx % max_points == 0]
                        if len(f[1:])==0:
                            sn.append([s[x][0] + start, s[x][1] + start])
                            #print("good segment",sn)
                        else:
                            #print("idx",idx,f[1:],len(f[1:]))
                            if leng-f[1:][-1]<10:
                                f[1:][-1]=leng

                            d = list(zip(f[:-1], f[1:]))
                            if f[1:][-1]!=leng:
                                d.append((f[1:][-1],leng))
                            subtrajectories = {}
                            for i in d:
                                #print(i)
                                #s[i] =
                                sn.append([i[0] + start2, i[1] + start2])
                             #   print(j,i[0] + start2, i[1] + start2)
                                j=j+1
                                #subtrajectories[i] = self.row_data.iloc[i[0]:i[1], :]
                            del f
                            del idx
                    else:

                        #sn[j] = (s[x][0] + start, s[x][1] + start)
                        sn.append([s[x][0] + start, s[x][1] + start])
                        j=j+1
                    #print(sn)
                # d=map(lambda (x, y):(x + start, y + start), s)
                #  print "start:",start,"s",s,"d:",d
                new_segments.extend(sn)
            #    st=nst
            # print "ddd",end, endd
            # new_segments.append([end,endd])
            # print(new_segments)
            # print end,endd

            segments = new_segments
        # s=[end,endd]

        # print segments
        trajectorySegments = {}
        # print len(segments)-1
        for seg in range(len(segments)):
            start = segments[seg][0]
            end = segments[seg][1]
            trajectorySegments[seg] = self.row_data.iloc[start:end, :]
        # seg=5682
        # print nst[0],len(nst)
        # print seg,nst[seg]
        return segments, trajectorySegments;

    def onelabelsegmentation(self, df, label='transportation_mode'):
        t = df[label].values
        start = 0
        segments = []
        for i in range(len(t) - 1):
            if (t[i] != t[i + 1]):
                #      print "ff:",start,i+1
                segments.append([start, i + 1])
                start = i + 1

        # print "ddddd",start,len(t)
        segments.append([start, len(t)])

        trajectorySegments = {}
        i = 0
        for segment in segments:
            trajectorySegments[i] = df.iloc[segment[0]:segment[1], :]
            i = i + 1
        del df
        del t
        return segments, trajectorySegments

    def segmentByLabel(self, label='transportation_mode'):
        df = self.row_data
        df = df.sort_values(by=[label])
        t = df[label].values
        start = 0
        segments = []
        for i in range(len(t) - 1):
            if (t[i] != t[i + 1]):
                segments.append([start, i + 1])
                start = i + 1

        trajectorySegments = {}
        i = 0
        for segment in segments:
            trajectorySegments[i] = df.iloc[segment[0]:segment[1], :]
            i = i + 1
        del df
        del t
        return segments, trajectorySegments

    def segmentByTime(self, meanTime=None, minimumNumberOfitemsInTraj=10):
        df = self.row_data
        t1 = pd.to_datetime(df.index)
        t2 = t1[1:]
        t1 = t1[:-1]
        sec = (t2 - t1).seconds
        if meanTime == None:
            meanTime = sec.mean()
        g = np.where(sec > meanTime)
        startSegment = 0
        segments = []
        for i in range(len(g[0])):
            if (g[0][i] - startSegment >= minimumNumberOfitemsInTraj):
                segments.append([startSegment, g[0][i]])
                startSegment = g[0][i] + 1
        trajectorySegments = {}
        i = 0
        for segment in segments:
            trajectorySegments[i] = self.row_data.iloc[segment[0]:segment[1], :]
            i = i + 1
        del df
        del t1
        del t2
        del sec
        return segments, trajectorySegments

    def __del__(self):
        del self.row_data
        print('clear memory')
