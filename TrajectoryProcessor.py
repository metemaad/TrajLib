import threading

import pandas as pd

import Trajectory as tr
import TrajectorySegmentation as trs


class TrajectoryProcessor:
    def __init__(self, ):
        self.processor = None
        self.target = ''
        self.labels = []
        self.results=[]

    def extract_features(self, seg, l, smooth=False):

        #try:

        t1 = tr.Trajectory(trajectory=seg, labels=[self.target], mood='df')
        ss = t1.rows()
        if ss > 10:
                # t1.get_features(smooth_=False)
                t1.get_features(smooth_=smooth)
                # t1.get_full_features()
                # print(t1.stat_label())
                extracted_features = t1.get_full_features()
                # print(extracted_features)
                self.results.append(extracted_features)
                target_lbl = str(t1.descriptor.target_label)
                nstr_speed=t1.process_speed_sub_traj()
                nstr_acc= t1.process_acc_sub_traj()
                nstr_bearing =t1.process_bearing_sub_traj()
                self.traj_txts_speed =self.traj_txts_speed+" "+nstr_speed
                self.traj_txts_acc = self.traj_txts_acc + " " + nstr_acc
                self.traj_txts_bearing = self.traj_txts_bearing + " " + nstr_bearing
                self.traj_train_txts_speed=self.traj_train_txts_speed+" "+nstr_speed+","+target_lbl+'\n'
                self.traj_train_txts_acc = self.traj_train_txts_acc + " " + nstr_acc + "," + target_lbl + '\n'
                self.traj_train_txts_bearing = self.traj_train_txts_bearing + " " + nstr_bearing + "," + target_lbl + '\n'

                print("processing ", l, "traj:", seg.shape,target_lbl,  extracted_features)

        del t1


        #except:
        #    print("Unexpected error:", sys.exc_info()[0])

    def geo_data_to_feature(self, src_='/home/mohammadetemad/geolife_full_labels.csv', target_='transportation_mode',
                            labels_=['day', 't_user_id', 'transportation_mode'], limit=2000, lat_='latitude',
                            lon_='longitude', alt_=None, time_date_='collected_time', seperator_=',',
                            output_file_name='features.csv'):
        self.results=[]
        ts = trs.TrajectorySegmentation()
        self.labels = labels_
        self.target = target_
        self.traj_txts_speed=""
        self.traj_train_txts_speed=""
        self.traj_txts_acc = ""
        self.traj_train_txts_acc = ""
        self.traj_txts_bearing = ""
        self.traj_train_txts_bearing = ""
        ts.load_data(lat=lat_, lon=lon_, alt=alt_, time_date=time_date_,labels=labels_, src=src_, seperator=seperator_)
        segments, trajectorySegments = ts.multi_label_segmentation(labels=labels_)
        threads = [None] * len(segments)

        for i in range(len(segments)):
            print("start", i, "of", len(segments))
            t = threading.Thread(target=self.extract_features, args=(trajectorySegments[i], i))
            try:
                t.start()
                threads[i] = t
            except:
                if threads:
                    threads[i - 1].join()
                    # del threads[0]

        for t in threads:
            t.join()
            del t
        print("all joined", len(self.results))
        print("done")
        #np.savetxt('Output.txt', self.traj_txts, fmt='%s')
        print("txt:")
        #print(len(self.traj_txts), self.traj_txts)

        f = open('output_speed.txt', 'w')
        f.write(self.traj_txts_speed)
        f.close()
        f = open('output_acc.txt', 'w')
        f.write(self.traj_txts_acc)
        f.close()
        f = open('output_bearing.txt', 'w')
        f.write(self.traj_txts_bearing)
        f.close()

        f = open('train_speed.csv', 'w')
        f.write(self.traj_train_txts_speed)
        f.close()
        f = open('train_acc.csv', 'w')
        f.write(self.traj_train_txts_acc)
        f.close()
        f = open('train_bearing.csv', 'w')
        f.write(self.traj_train_txts_bearing)
        f.close()

        print("txt")

        cols = [u'distance_min_', u'distance_max_', u'distance_mean',
                u'distance_median', u'distance_std', u'distance_p10', u'distance_p25',
                u'distance_p50', u'distance_p75', u'distance_p90', u'speed_min_',
                u'speed_max_', u'speed_mean', u'speed_median', u'speed_std',
                u'speed_p10', u'speed_p25', u'speed_p50', u'speed_p75', u'speed_p90',
                u'acc_min_', u'acc_max_', u'acc_mean', u'acc_median', u'acc_std',
                u'acc_p10', u'acc_p25', u'acc_p50', u'acc_p75', u'acc_p90',
                u'bearing_min_', u'bearing_max_', u'bearing_mean', u'bearing_median',
                u'bearing_std', u'bearing_p10', u'bearing_p25', u'bearing_p50',
                u'bearing_p75', u'bearing_p90', u'jerk_min_', u'jerk_max_',
                u'jerk_mean', u'jerk_median', u'jerk_std', u'jerk_p10', u'jerk_p25',
                u'jerk_p50', u'jerk_p75', u'jerk_p90', u'brate_min_', u'brate_max_',
                u'brate_mean', u'brate_median', u'brate_std', u'brate_p10',
                u'brate_p25', u'brate_p50', u'brate_p75', u'brate_p90',
                u'brate_rate_min_', u'brate_rate_max_', u'brate_rate_mean',
                u'brate_rate_median', u'brate_rate_std', u'brate_rate_p10',
                u'brate_rate_p25', u'brate_rate_p50', u'brate_rate_p75',
                u'brate_rate_p90', u'isInValid', u'isPure', u'target']
        features = pd.DataFrame(self.results, columns=cols)
        features.to_csv(output_file_name)
        return features
    def geo_data_to_feature_no_multithreading(self, src_='/home/mohammadetemad/geolife_full_labels.csv', target_='transportation_mode',
                            labels_=['day', 't_user_id', 'transportation_mode'], limit=2000, lat_='latitude',
                            lon_='longitude', alt_=None, time_date_='collected_time', seperator_=',',
                            output_file_name='features.csv',
                            output_file_speed_train='output_speed.txt',output_file_speed_context='train_speed.csv',
                            output_file_acc_train='output_acc.txt',output_file_acc_context='train_acc.csv',
                            output_file_bearing_train = 'output_bearing.txt', output_file_bearing_context = 'train_bearing.csv'):
        self.results=[]
        ts = trs.TrajectorySegmentation()
        self.labels = labels_
        self.target = target_
        self.traj_txts_speed = ""
        self.traj_train_txts_speed = ""
        self.traj_txts_acc = ""
        self.traj_train_txts_acc = ""
        self.traj_txts_bearing = ""
        self.traj_train_txts_bearing = ""
        ts.load_data(lat=lat_, lon=lon_, alt=alt_, time_date=time_date_,labels=labels_, src=src_, seperator=seperator_)
        segments, trajectorySegments = ts.multi_label_segmentation(labels=labels_)
        #threads = [None] * len(segments)

        for i in range(len(segments)):

            print("start", i, "of", len(segments))
            self.extract_features(trajectorySegments[i], i)

        print("all joined", len(self.results))
        print("done")
        #np.savetxt('output.txt',self.traj_txts, fmt='%s')
        print("txt:")
        #print(len(self.traj_txts), self.traj_txts)
        f = open(output_file_speed_train, 'w')
        f.write(self.traj_txts_speed)
        f.close()
        f = open(output_file_acc_train, 'w')
        f.write(self.traj_txts_acc)
        f.close()
        f = open(output_file_bearing_train, 'w')
        f.write(self.traj_txts_bearing)
        f.close()

        f = open(output_file_speed_context, 'w')
        f.write(self.traj_train_txts_speed)
        f.close()
        f = open(output_file_acc_context, 'w')
        f.write(self.traj_train_txts_acc)
        f.close()
        f = open(output_file_bearing_context, 'w')
        f.write(self.traj_train_txts_bearing)
        f.close()


        print("txt")

        cols = [u'distance_min_', u'distance_max_', u'distance_mean',
                u'distance_median', u'distance_std', u'distance_p10', u'distance_p25',
                u'distance_p50', u'distance_p75', u'distance_p90', u'speed_min_',
                u'speed_max_', u'speed_mean', u'speed_median', u'speed_std',
                u'speed_p10', u'speed_p25', u'speed_p50', u'speed_p75', u'speed_p90',
                u'acc_min_', u'acc_max_', u'acc_mean', u'acc_median', u'acc_std',
                u'acc_p10', u'acc_p25', u'acc_p50', u'acc_p75', u'acc_p90',
                u'bearing_min_', u'bearing_max_', u'bearing_mean', u'bearing_median',
                u'bearing_std', u'bearing_p10', u'bearing_p25', u'bearing_p50',
                u'bearing_p75', u'bearing_p90', u'jerk_min_', u'jerk_max_',
                u'jerk_mean', u'jerk_median', u'jerk_std', u'jerk_p10', u'jerk_p25',
                u'jerk_p50', u'jerk_p75', u'jerk_p90', u'brate_min_', u'brate_max_',
                u'brate_mean', u'brate_median', u'brate_std', u'brate_p10',
                u'brate_p25', u'brate_p50', u'brate_p75', u'brate_p90',
                u'brate_rate_min_', u'brate_rate_max_', u'brate_rate_mean',
                u'brate_rate_median', u'brate_rate_std', u'brate_rate_p10',
                u'brate_rate_p25', u'brate_rate_p50', u'brate_rate_p75',
                u'brate_rate_p90', u'isInValid', u'isPure', u'target']
        features = pd.DataFrame(self.results, columns=cols)
        features.to_csv(output_file_name)
        return features

    def __del__(self):
        return None
