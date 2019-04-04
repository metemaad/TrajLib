import numpy as np


class Distances:

    @staticmethod
    def distance_array(lat, lon):
        lat2 = np.append(lat[1:], lat[-1:])
        lon2 = np.append(lon[1:], lon[-1:])
        # R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
        r = 6372.8
        d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
        return 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter

    @staticmethod
    def calculate_two_point_distance(lat, lon, lat2, lon2):
        r = 6372.8
        d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
        return 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter
