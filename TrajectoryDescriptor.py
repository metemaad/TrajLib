import numpy as np
import pandas as pd
import TrajectoryDescriptorFeature as tdf
from TrajectoryFeatureExtractor import TrajectoryFeatureExtractor


class TrajectoryDescriptor:
    def __init__(self, **kwargs):
        self.isInValid = False
        self.isPure = False
        self.row_data = kwargs.get('trajectory', pd.DataFrame())
        self.labels = kwargs.get('labels', ['target'])
        self.stop_parameters = kwargs.get('stop_parameters', [100, 60, 60, 100])
        if self.row_data.shape[0] == 0:
            self.isInValid = True
        self.purity_labels = self.purity()
        self.target_label = self.get_target()

    def get_target(self):
        if len(self.purity_labels) == 1:
            return list(self.purity_labels.keys())[0]
        else:
            print(self.row_data)
            sorted_dic = sorted(self.purity_labels, key=self.purity_labels.get, reverse=True)
            return list(sorted_dic.keys())[0]  #TODO descobrir erro

    def purity(self):
        label_dic = {}
        for ix, v in self.row_data.groupby(self.labels):
            label_dic[ix] = v.shape[0]
        if len(label_dic) == 1:
            self.isPure = True
        else:
            self.isPure = False
        return label_dic

    def describe(self):
        trajectory_descriptor_feature = tdf.TrajectoryDescriptorFeature()
        tfe = TrajectoryFeatureExtractor(trajectory=self.row_data, stop_parameters=self.stop_parameters)
        stops = tfe.get_stop_times()
        stops_rate = 0 if len(stops) == 1 and stops[0] == 0 else len(stops)

        td = trajectory_descriptor_feature.describe(self.row_data.td)
        trajectory_descriptor_feature.reset()
        other = [self.isInValid, self.isPure, self.target_label, stops_rate]
        other = other + tfe.get_list_of_features()
        distance = trajectory_descriptor_feature.describe(self.row_data.distance)

        trajectory_descriptor_feature.reset()
        speed = trajectory_descriptor_feature.describe(self.row_data.speed)

        trajectory_descriptor_feature.reset()
        acc = trajectory_descriptor_feature.describe(self.row_data.acc)

        trajectory_descriptor_feature.reset()
        bearing = trajectory_descriptor_feature.describe(self.row_data.bearing)

        trajectory_descriptor_feature.reset()
        jerk = trajectory_descriptor_feature.describe(self.row_data.jerk)

        trajectory_descriptor_feature.reset()
        brate = trajectory_descriptor_feature.describe(self.row_data.brate)

        trajectory_descriptor_feature.reset()
        brrate = trajectory_descriptor_feature.describe(self.row_data.brrate)

        trajectory_descriptor_feature.reset()
        stop_time = trajectory_descriptor_feature.describe(stops)

        ret = distance + speed + acc + bearing + jerk + brate + brrate + stop_time + other

        return ret

    def get_full_features_column_name(self):
        other = ['isInValid', 'isPure', 'target']

        features = np.array(['min_', 'max_', 'mean', 'median', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'])

        speed_features = np.array(['speed_'] * len(features))
        speed_features = map(''.join, zip(speed_features, features))

        distance_features = np.array(['distance_'] * len(features))
        distance_features = map(''.join, zip(distance_features, features))

        acc_features = np.array(['acc_'] * len(features))
        acc_features = map(''.join, zip(acc_features, features))

        bearing_features = np.array(['bearing_'] * len(features))
        bearing_features = map(''.join, zip(bearing_features, features))

        jerk_features = np.array(['jerk_'] * len(features))
        jerk_features = map(''.join, zip(jerk_features, features))

        brate_features = np.array(['brate_'] * len(features))
        brate_features = map(''.join, zip(brate_features, features))

        brate_rate__features = np.array(['brate_rate_'] * len(features))
        brate_rate__features = map(''.join, zip(brate_rate__features, features))


        ret = map(''.join, zip(distance_features, speed_features, acc_features, bearing_features,
                               jerk_features, brate_features, brate_rate__features, other))

        return ret
