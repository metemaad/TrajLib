import numpy as np


class TrajectoryDescriptorFeature:
    def __init__(self):
        self.min_, self.max_, self.mean, self.median, self.std, self.p10, self.p25, self.p50, self.p75, self.p90 = np.zeros(
            10)

    def reset(self):
        self.min_, self.max_, self.mean, self.median, self.std, self.p10, self.p25, self.p50, self.p75, self.p90 = np.zeros(
            10)
        return 0

    def describe(self, trajectory_feature):
        self.min_ = np.min(trajectory_feature)
        self.max_ = np.max(trajectory_feature.values)
        self.mean = np.mean(trajectory_feature)
        self.median = np.median(trajectory_feature)
        self.std = np.std(trajectory_feature)
        self.p10 = np.percentile(trajectory_feature, 10)
        self.p25 = np.percentile(trajectory_feature, 25)
        self.p50 = np.percentile(trajectory_feature, 50)
        self.p75 = np.percentile(trajectory_feature, 75)
        self.p90 = np.percentile(trajectory_feature, 90)
        return [self.min_, self.max_, self.mean, self.median, self.std, self.p10, self.p25, self.p50, self.p75,
                self.p90]
