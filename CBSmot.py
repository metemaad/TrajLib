import Distances as d
import pandas as pd
# import numpy as np


class CBSmot:

    def count_neighbors(self, traj, position, max_dist):
        neighbors = 0
        yet = True
        j = position + 1
        while j < len(traj.index) and yet:
            if d.Distances.calculate_two_point_distance(traj.iloc[position]['lat'],
                                                        traj.iloc[position]['lon'],
                                                        traj.iloc[j]['lat'],
                                                        traj.iloc[j]['lon']) < max_dist:
                neighbors += 1
            else:
                yet = False
            j += 1
        return neighbors

    def centroid(self, subtraj):
        x = 0
        y = 0

        for index, row in subtraj.iterrows():
            x += row['lat']
            y += row['lon']

        return [x/len(subtraj.index), y/len(subtraj.index)]

    def clean_stops(self, stops, min_time):
        stops_aux = stops.copy()
        for stop in stops:
            p1 = stop.index.values[0]
            p2 = stop.index.values[-1]
            if (p2 - p1).item() / 1000000000 < min_time:
                stops_aux.remove(stop)
        return stops_aux

    def clean_stops_segment(self, stops, min_time, index):
        stops_aux = stops.copy()
        i = 0
        for stop in stops:
            p1 = stop.index.values[0]
            p2 = stop.index.values[-1]
            if (p2 - p1).item() / 1000000000 < min_time:
                stops_aux.remove(stop)
                index.pop(i)
            i += 1
        return index, stops_aux

    def merge_stop(self, stops, max_dist, time_tolerance):
        i = 0
        while i < len(stops):
            if (i+1) < len(stops):
                s1 = stops[i]
                s2 = stops[i+1]
                p2 = s2.index.values[0]
                p1 = s1.index.values[-1]
                if (p2 - p1).item() / 1000000000 <= time_tolerance:
                    c1 = self.centroid(s1)
                    c2 = self.centroid(s2)
                    if d.Distances.calculate_two_point_distance(c1[0], c1[1], c2[0], c2[1]) <= max_dist:
                        stops.pop(i+1)
                        s1.append(s2, ignore_index=True)
                        stops[i] = s1
                        i -= 1
            i += 1
        return stops

    def merge_stop_segment(self, stops, max_dist, time_tolerance, index):
        i = 0
        while i < len(stops):
            if (i+1) < len(stops):
                s1 = stops[i]
                s2 = stops[i+1]
                p2 = s2.index.values[0]
                p1 = s1.index.values[-1]
                if (p2 - p1).item() / 1000000000 <= time_tolerance:
                    c1 = self.centroid(s1)
                    c2 = self.centroid(s2)
                    if d.Distances.calculate_two_point_distance(c1[0], c1[1], c2[0], c2[1]) <= max_dist:
                        index_i = index[i]
                        index_i_1 = index[i+1]
                        stops.pop(i+1)
                        index.pop(i+1)
                        s1.append(s2, ignore_index=True)
                        stops[i] = s1
                        index[i] = [index_i[0], index_i_1[-1]]
                        i -= 1
            i += 1
        return index, stops

    def find_stops(self, traj, max_dist, min_time, time_tolerance, merge_tolerance):
        neighborhood = [0]*len(traj.index)
        stops = []
        traj.sort_index(inplace=True)

        j = 0
        while j < len(traj.index):
            valor = self.count_neighbors(traj, j, max_dist)
            neighborhood[j] = valor
            j += valor
            j += 1

        for i in range(len(neighborhood)):
            if neighborhood[i] > 0:
                p1 = pd.to_datetime(traj.iloc[i].name)
                p2 = pd.to_datetime(traj.iloc[i + neighborhood[i]-1].name)
                diff = (p2 - p1).total_seconds()
                if diff >= time_tolerance:
                    stops.append(traj.loc[p1:p2])

        stops = self.merge_stop(stops, max_dist, merge_tolerance)
        stops = self.clean_stops(stops, min_time)
        return stops

    def segment_stops_moves(self, traj, max_dist, min_time, time_tolerance, merge_tolerance):
        neighborhood = [0]*len(traj.index)
        stops = []
        index = []
        traj.sort_index(inplace=True)

        j = 0
        while j < len(traj.index):
            valor = self.count_neighbors(traj, j, max_dist)
            neighborhood[j] = valor
            j += valor
            j += 1

        for i in range(len(neighborhood)):
            if neighborhood[i] > 0:
                p1 = pd.to_datetime(traj.iloc[i].name)
                p2 = pd.to_datetime(traj.iloc[i + neighborhood[i]-1].name)
                diff = (p2 - p1).total_seconds()
                if diff >= time_tolerance:
                    stops.append(traj.loc[p1:p2])
                    index.append([p1, p2])

        index, stops = self.merge_stop_segment(stops, max_dist, merge_tolerance, index)
        index, stops = self.clean_stops_segment(stops, min_time, index)
        return index, stops

