import numpy as np
import pandas as pd
from Distances import Distances
from collections import namedtuple
from CBSmot import CBSmot


class TrajectoryFeatureExtractor:

    def __init__(self, **kwargs):
        data = kwargs.get('trajectory', pd.DataFrame())
        data.sort_index(inplace=True)
        self.data = data
        stop_parameters = kwargs.get('stop_parameters', [100, 60, 60, 100])
        cbsmote = CBSmot()
        self.stops = cbsmote.find_stops(self.data, stop_parameters[0], stop_parameters[1], stop_parameters[2],
                                        stop_parameters[3])

    def get_start_time(self):
        return self.data.index[0]

    def get_end_time(self):
        return self.data.index[-1]

    def get_start_lat(self):
        return self.data.lat[0]

    def get_end_lat(self):
        return self.data.lat[-1]

    def get_start_lon(self):
        return self.data.lon[0]

    def get_end_lon(self):
        return self.data.lon[-1]

    def get_is_week_day(self):
        return 1 if self.data.index[0].weekday() < 5 else 0

    def get_day_of_week(self):
        return self.data.index[0].weekday()

    def get_duration_in_second(self):
        p1 = self.data.index.values[0]
        p2 = self.data.index.values[-1]
        return (p2 - p1).item() / 1000000000

    def get_distance_travelled(self):
        return np.sum(self.data.distance)

    def get_start_end_distance(self):
        return Distances.calculate_two_point_distance(self.get_start_lat(), self.get_start_lon(), self.get_end_lat(),
                                                      self.get_end_lon())

    def get_number_self_intersect(self):
        Coordinate = namedtuple("Coordinate", ["lat", "long"])
        coordinates = {}
        count = 0
        for index, row in self.data.iterrows():
            local = Coordinate(lat=row['lat'], long=row['lon'])
            if coordinates.keys().__contains__(local):
                count += 1
            else:
                coordinates[local] = 1
        return count

    def get_distance_by_day(self, day):
        distances = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for index, row in self.data.iterrows():
            date = index.weekday()
            distances[date] = distances.get(date) + row['distance']
        return list(distances.values())[day]

    def get_stop_times(self):
        stop_times = []
        for i in range(len(self.stops)):
            t1 = self.stops[i].index.values[0]
            t2 = self.stops[i].index.values[-1]
            stop_times.append((t2-t1).item()/ 1000000000)
        return pd.Series(stop_times) if len(stop_times) > 0 else pd.Series([0])

    def get_stop_total(self):
        return np.sum(self.get_stop_times())

    def get_stop_over_total(self):
        if self.get_duration_in_second() == 0:
            return 0
        return self.get_stop_total() / self.get_duration_in_second()

    def get_list_of_features(self):
        return [self.get_start_time(), self.get_end_time(), self.get_is_week_day(), self.get_day_of_week(),
                self.get_duration_in_second(), self.get_distance_travelled(), self.get_start_end_distance(),
                self.get_start_lat(), self.get_start_lon(), self.get_end_lat(), self.get_end_lon(),
                self.get_number_self_intersect(), self.get_distance_by_day(0), self.get_distance_by_day(1),
                self.get_distance_by_day(2), self.get_distance_by_day(3), self.get_distance_by_day(4),
                self.get_distance_by_day(5), self.get_distance_by_day(6), self.get_stop_total(),
                self.get_stop_over_total()]
