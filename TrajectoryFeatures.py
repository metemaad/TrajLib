import random

import numpy as np
import pandas as pd
import pywt as pywavelets
from scipy.signal import lfilter
import Distances as d


def mad(a):
    c = 0.67448975019608171
    axis = 0
    center = np.median
    center = np.apply_over_axes(center, a, axis)
    return np.median((np.fabs(a - center)) / c, axis=axis)


def wavelet_smoother(x, wavelet="db4", level=1, title=None):
    coeff = pywavelets.wavedec(x, wavelet, mode="per")
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywavelets.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    return pywavelets.waverec(coeff, wavelet, mode="per")


class TrajectoryFeatures:
    def __init__(self, **kwargs):

        self.row_data = kwargs.get('trajectory', pd.DataFrame())
        self.labels = kwargs.get('labels', ['target'])
        self.smooth_ = kwargs.get('smooth', False)
        self.sgn_ = kwargs.get('sgn', False)

        self.get_duration()  # 1

        self.get_distance(smooth=False)  # 2

        self.get_speed(smooth=self.smooth_)  # 3

        self.get_acc(smooth=self.smooth_, sgn=self.sgn_)  # 4

        self.get_bearing(smooth=self.smooth_)  # 5

        self.get_jerk(smooth=self.smooth_, sgn=self.sgn_)  # 6

        self.get_brate(smooth=self.smooth_)  # 7

        self.get_brrate(smooth=self.smooth_)  # 8



    def smoother(self, signal, n=100, a=1, level1=0, level2=0, plot=False):
        b = [1.0 / n] * n
        yy = lfilter(b, a, signal)
        #        print(signal.shape)
        #        print(yy.shape)

        # yff = filtfilt(b, a, signal,method="gust")
        #        yff = yy

        #        y8 = self.wavelet_smoother(signal, level=level1)
        #        if len(y8) != len(signal):
        #            y8 = y8[0:len(signal)]
        #        ym = (y8 + yy + yff) / 3
        #        ymm = self.wavelet_smoother(ym, level=level2)

        #        if len(ymm) != len(signal):
        #            ymm = ymm[0:len(signal)]
        #        if plot:
        # plt.plot(ymm, linewidth=2, linestyle="-", c="b")
        # print(ymm.shape)
        # plt.show()
        #            print("plot")
        return yy

    def get_duration(self):
        t = np.diff(pd.to_datetime(self.row_data.index)) / 1000000000  # convert to second
        t = t.astype(np.float64)
        t = np.append(t[0:], t[-1:])
        tmp = self.row_data.assign(td=t)
        tmp1 = tmp.loc[tmp['td'] > 0, :]
        # avoid NaN in case rate of sampling is more than 1 per second
        self.row_data = tmp1
        self.row_data.assign(timestamp=self.row_data.index)
        del tmp
        del tmp1
        return t

    def get_theta(self):
        self.polartheta = np.arctan(self.row_data['lon'] / self.row_data['lat'])
        if 'theta' in self.row_data.columns:
            self.row_data.assign(theta=self.polartheta)
        else:
            self.row_data.assign(theta=self.polartheta)
        return self.polartheta

    def get_r(self):
        self.polarR = np.sqrt(self.row_data['lon'] ** 2 + self.row_data['lat'] ** 2)
        self.row_data.assign(R=self.R)

        return self.polarR

    def get_distance(self, smooth=False):
        lat = self.row_data.lat.values
        lon = self.row_data.lon.values
        distance_val = d.Distances.distance_array(lat, lon)
        # this is the distance difference between two points not the Total distance traveled
        if smooth:
            distance_val = self.smoother(distance_val)
        self.row_data = self.row_data.assign(distance=distance_val)

        return distance_val

    """calculate speed"""

    def get_speed(self, smooth=False):
        speed_val = self.row_data.distance / self.row_data.td
        if np.isnan(speed_val).any():
            print("error1")
            a = random.getrandbits(128)
            print(a)
            self.row_data.to_csv(str(a) + '.csv')
        if smooth:
            speed_val = self.smoother(speed_val)

        self.row_data = self.row_data.assign(speed=speed_val)
        if np.isnan(speed_val).any():
            print("error2")
            a = random.getrandbits(128)
            print(a)
            self.row_data.to_csv(str(a) + '.csv')
        return speed_val

    """calculate acc"""

    def get_acc(self, smooth=False, sgn=False):
        _speed_diff = np.diff(self.row_data.speed)
        _speed_diff = np.append(_speed_diff, _speed_diff[-1:])
        acc_val = _speed_diff / self.row_data.td
        if sgn:
            acc_val = np.sign(acc_val)
        if smooth:
            acc_val = self.smoother(acc_val)
        self.row_data = self.row_data.assign(acc=acc_val)

        return acc_val

    """calculate bearing"""

    def get_bearing(self, smooth=False):
        lat = self.row_data.lat.values
        lon = self.row_data.lon.values
        lat2 = np.append(lat[1:], lat[-1:])
        lon2 = np.append(lon[1:], lon[-1:])

        lat1, lat2, diff_long = map(np.radians, (lat, lat2, lon2 - lon))
        a = np.sin(diff_long) * np.cos(lat2)
        b = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
        bearing_val = np.arctan2(a, b)
        bearing_val = np.degrees(bearing_val)
        bearing_val = (bearing_val + 360) % 360
        if smooth:
            bearing_val = self.smoother(bearing_val)
        self.row_data = self.row_data.assign(bearing=bearing_val)
        return bearing_val

    """calculate jerk"""

    def get_jerk(self, smooth=False, sgn=False):
        accdiff = np.diff(self.row_data.acc)
        accdiff = np.append(accdiff, accdiff[-1:])
        jerk_val = accdiff / self.row_data.td
        if sgn:
            jerk_val = np.sign(jerk_val)
        if smooth:
            jerk_val = self.smoother(jerk_val)
        self.row_data = self.row_data.assign(jerk=jerk_val)
        return jerk_val

    """calculate brate"""

    def get_brate(self, smooth=False):
        compass_bearingdiff = np.diff(self.row_data.bearing)
        compass_bearingdiff = np.append(compass_bearingdiff, compass_bearingdiff[-1:])
        brate_val = compass_bearingdiff / self.row_data.td
        if smooth:
            brate_val = self.smoother(brate_val)
        self.row_data = self.row_data.assign(brate=brate_val)
        return brate_val

    """calculate brrate"""

    def get_brrate(self, smooth=False):
        bratediff = np.diff(self.row_data.brate)
        bratediff = np.append(bratediff, bratediff[-1:])
        brrate_val = bratediff / self.row_data.td
        if smooth:
            brrate_val = self.smoother(brrate_val)
        self.row_data = self.row_data.assign(brrate=brrate_val)
        return brrate_val

