import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal
import matplotlib.pyplot as plt

from enum import Enum
from typing import List, Iterable, Tuple, Any, Optional


# TODO: Filter the data so that outliers are deleted from rest data and are flagged in repetition data
# TODO: Make a parser to automatically check the repetitions

class NormOption(Enum):
    """
    This can be safely ignored. It defines different ways to normalize data,
    but only one is currently used.
    """
    RestAvg = 0
    RestNDist = 1
    TotAvg = 2
    RunAvg = 3


class FeatureType(Enum):
    """
    These enums tell the algorithm what value to compute
    """
    DIST = 0
    POS = 1
    AREA = 2


class FeatureLandmarks(Enum):
    pass


# # These enums define which landmarks to use for each metric type
# class StrokeFeatureLandmarks(FeatureLandmarks):
#     MouthHeight = [51, 57]  # O_{Max} and O_{Min}
#     MouthWidth = [48, 54]  # W_{Max} and W_{Min}
#     LeftMouth = [51, 54, 57]  # A_{diff}
#     RightMouth = [51, 48, 57]  # A_{diff}
#     Corners = [48, 54, 30]  # r_{LCRC}
#     LEyebrowNasion = [24, 30]  # d_0{diff} Left side  // changed 42 for 30 | 23 for 24
#     REyebrowNasion = [19, 30]  # d_0{diff} Right side // changed 39 for 30 | 20 for 19
#     LCanthusMouthC = [42, 54]  # d_1{diff} Left side
#     RCanthusMouthC = [39, 48]  # d_1{diff} Right side
#     LCanthusMouthU = [42, 51]  # d_2{diff} Right side       P.S. There's a
#     RCanthusMouthU = [39, 51]  # d_2{diff} Left side        typo in the paper
#     LMouthCMouthL = [54, 57]  # d_3{diff} Right side // changed 51 for 57
#     RMouthCMouthL = [48, 57]  # d_3{diff} Left side // changed 51 for 57

# These enums define which landmarks to use for each metric type



class StrokeFeatureLandmarks(FeatureLandmarks):
    MouthHeight = [51, 57]  # O_{Max} and O_{Min}
    MouthWidth = [48, 54]  # W_{Max} and W_{Min}
    LeftMouth = [51, 54, 57]  # A_{diff}
    RightMouth = [51, 48, 57]  # A_{diff}
    Corners = [48, 54, 30]  # r_{LCRC}
    NoseMouthR = [30, 48] # Nose to right corner of the mouth
    NoseMouthL = [30, 54] # Nose to left corner of the mouth
    LEyebrowCanthus = [23, 42]  # d_0{diff} Left side
    REyebrowCanthus = [20, 39]  # d_0{diff} Right side
    LCanthusMouthC = [42, 54]  # d_1{diff} Left side
    RCanthusMouthC = [39, 48]  # d_1{diff} Right side
    LCanthusMouthU = [42, 51]  # d_2{diff} Right side       P.S. There's a
    RCanthusMouthU = [39, 51]  # d_2{diff} Left side        typo in the paper
    LMouthCMouthU = [54, 51]  # d_3{diff} Right side
    RMouthCMouthU = [48, 51]  # d_3{diff} Left side

class ALSFeatureLandmarks(FeatureLandmarks):
    LowerLip = [30, 57]  # LLpath sum and vLL_Max and vLL_Min
    MouthAreaLeft = [54, 51, 57]  # A_Mean and delta_A and A_Absdiff
    MouthAreaRight = [48, 51, 57]  # A_Mean and delta_A and A_Absdiff
    MouthWidth = [48, 54]  # vW_Max and vW_Min and e_Mean
    MouthHeight = [51, 57]  # e_Mean
    Corners = [48, 54, 30]  # r_Symm
    NoseMouthR = [30, 48]  # Nose to right corner of the mouth
    NoseMouthL = [30, 54]  # Nose to left corner of the mouth

class FeatureLandmarks(FeatureLandmarks):
    MouthHeight = [51, 57]  # O_{Max} and O_{Min}
    MouthWidth = [48, 54]  # W_{Max} and W_{Min}
    MouthAreaLeft = [54, 51, 57]  # A_Mean and delta_A and A_Absdiff
    MouthAreaRight = [48, 51, 57]  # A_Mean and delta_A and A_Absdiff
    Corners = [48, 54, 30]  # r_{LCRC}
    NoseLipR = [30, 48] # Nose to right corner of the mouth
    NoseLipL = [30, 54] # Nose to left corner of the mouth
    NoseJawR = [30, 48] # Nose to right jaw
    NoseJawL = [30, 54] # Nose to left jaw
    LEyebrowCanthus = [23, 42]  # d_0{diff} Left side
    REyebrowCanthus = [20, 39]  # d_0{diff} Right side
    LCanthusMouthC = [42, 54]  # d_1{diff} Left side
    RCanthusMouthC = [39, 48]  # d_1{diff} Right side
    LCanthusMouthU = [42, 51]  # d_2{diff} Right side       P.S. There's a
    RCanthusMouthU = [39, 51]  # d_2{diff} Left side        typo in the paper
    LMouthCMouthU = [54, 51]  # d_3{diff} Right side
    RMouthCMouthU = [48, 51]  # d_3{diff} Left side
    LowerLip = [30, 57]  # LLpath sum and vLL_Max and vLL_Min

# We have some options about what column names may be depending on if the data
# is 3D or 2D
default_columns = [
    "landmark_{}_x",
    "landmark_{}_y",
    "landmark_{}",
    "landmark_{}.1",
    "landmark_{}.2"
]

class Metrics:
    """
    General Outline of Processing:
    1. Init(Compute preliminary values) -
        a. The dimension of the data
        b. Data filtering coefficients
    2. Evaluate Feature(Get values for the given feature over each frame) -
        a. Inputs:
            i. A FeatureLandmarks object that points to an array of landmark
            indexes. These are the points that actually define which landmarks
            make up the feature in question
            ii. A FeatureType object that defines the algorithm used to compute
            the feature. The options are Distance, Position, or Area
        b. Call columns_to_numpy():
            This gets the position values for each landmark in the
            FeatureLandmark object over the entire video.
            This step also includes filtering using a low pass and median filter
        c. Compute the feature value:
            General Flow:
                i. Compute the Distance or Area for each frame for both active and rest frames
                ii. Compute a mask that removes outliers from the rest data
            If the feature is Position:
                Then there is nothing left to do. We just return the data
    3. get_{some metric}_metrics(Uses feature to compute the metric value):
        A metric is some number that defines the feature over the active period.
        Once you have the feature array, you can use some operation like a sum
        or average to reduce the feature array to one value
        a. Evaluate the feature as in step 2.
        b. Do some math... It varies for every metric.
            Sum the distances to get a path length
            Average them to get an average distance
            Average a three point difference to get average velocity
            etc
    """

    _landmarks: pd.DataFrame  # Holds position data for landmarks
    _fil_coeffs: np.ndarray  # Holds the coefficients used for the low pass filter
    _rest_frames: Iterable[int]  # Defines frame to normalize metrics on
    _res_landmarks: pd.DataFrame  # Holds the landmarks for the rest video
    _active_frames: Iterable[int]  # Defines frames to use for metrics
    _column_names: List[str]  # Defines the columns that positional data is stored in
    _dimension: int  # The dimension of the landmarks
    _win_length: int = 5  # The window to use for median and low pass filtering

    def __init__(self, landmarks: pd.DataFrame, rest_frame: pd.DataFrame, active_frames: Iterable[int] = None,
                 column_names=None):
        if column_names is None:
            column_names = default_columns.copy()
        self._landmarks = landmarks
        self._rest_landmarks = rest_frame
        self._active_frames = range(0, 0) if active_frames is None else active_frames
        self._dimension = 0
        # Different dimensions have different column names so we iterate over the column names to find the true dimension
        for column in column_names:
            if column.format(0) in self._landmarks.columns:
                self._dimension += 1
        if self._dimension == 3:
            self._landmarks = self._landmarks.drop(self._landmarks.index[
                                                       0])  # this line drops the first row of the 3d landmakrs, it contains only x, y, z index | Diego 4/12/2020
            self._rest_landmarks = self._rest_landmarks.drop(self._rest_landmarks.index[
                                                                 0])  # this line drops the first row of the 3d landmakrs, it contains only x, y, z index | Diego 4/12/2020

        self._column_names = column_names[:2] if self._dimension == 2 else column_names[2:]
        # if "Video_Frame_number" not in self._landmarks.columns:
        #     raise Exception(f"Video_Frame_number must be a column in your input data")
        if self._dimension not in [2, 3]:
            raise Exception(f"Dimension of data must be 2 or 3. Check that the column names are correct ({', '.join(column_names)})")
        if "Time_Stamp (s)" in self._landmarks.columns:
            self._fil_coeffs = self.create_filter(self._landmarks["Time_Stamp (s)"].to_numpy(), 12.5, 4)

    def get_feature_gradient(self, feature: np.ndarray) -> np.ndarray:
        """
        Takes the gradient of the feature array to get the gradient for
        estimating velocity or acceleration
        :param feature: An array of feature measures
        :return: The gradient of the measures
        """
        grads = [np.gradient(feature[:, dim]) for dim in range(self._dimension)]
        return np.column_stack(grads)

    def get_feature_trajectory(self, feature: np.ndarray) -> np.ndarray:
        """
        Finds the direction of the gradient
        :param feature: An array of feature measures
        :return: The trajectory of the measures
        """
        if self._dimension == 3:
            raise Exception("Trajectory not implemented for 3D positions")
        elif self._dimension == 2:
            gradients = self.get_feature_gradient(feature)
            return np.arctan2(gradients[:, 0], gradients[:, 1])
        else:
            raise Exception("Dimension of data must be 2D")

    @staticmethod
    def concordance_correlation_coefficient(s1, s2, remove_bias=False):

        if remove_bias:
            # adding this to test 4/25/2020
            # the idea is that by removing the bias is possible to compare only the variability of the signals
            s1 = s1 - np.mean(s1)
            s2 = s2 - np.mean(s2)

        N1 = len(s1)
        N2 = len(s2)
        if N1 == N2:
            N = N1
        elif N1 > N2:
            s1 = s1[0:N2]
            N = N2
        elif N1 < N2:
            s2 = s2[0:N1]
            N = N1
        m_s1 = np.mean(s1)
        m_s2 = np.mean(s2)
        s1_nomean = s1 - m_s1
        s2_nomean = s2 - m_s2
        s1_ss2 = (1 / N) * np.sum(s1_nomean**2)
        s2_ss2 = (1 / N) * np.sum(s2_nomean**2)
        s1s2 = (1 / N) * np.sum(np.multiply(s1_nomean,
                                            s2_nomean))  # np.multiply(A,B) -> element wise multiplication between matrices A and B
        p = (2 * s1s2) / (s1_ss2 + s2_ss2 + (m_s1 - m_s2)**2)
        return p

    @staticmethod
    def adjust_amplitude_and_time(sig, time, normalize=True, des_n=None):
        """
        Takes a signal x of length n and return a new signal x_n of length des_n and with zero mean and unit standard deviation

        Interpolation is performed using cubic splines
        """
        sig = sig[:, None]
        time = time[:, None]
        if des_n is None:
            des_time = time
        else:
            des_time = np.linspace(time[0], time[-1], des_n)
        if normalize:
            sig = (sig - np.mean(sig)) / np.std(sig)
        try:
            tck = interpolate.splrep(time, sig, s=0)
        except:
            tck = interpolate.splrep(np.sort(time, axis=None), sig, s=0)
        new_sig = interpolate.splev(des_time, tck, der=0)
        new_sig_der = interpolate.splev(des_time, tck, der=1)
        return new_sig, des_time, new_sig_der

    @staticmethod
    def poly_area_2d(frame) -> float:
        """
        Uses the shoelace formula to calculate area of a general 2D polygon
        :param frame: An (n, 2) matrix giving cartesian values for points
        :return: The area of the polygon
        """
        x, y = frame[:, 0], frame[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def triangle_3d_area(frame):
        """
        Uses the formula A=(1/2)V X U to calculate the area of a triangle
        :param frame: An (n, 3) matrix giving cartesian values for points
        :return: The area of the triangle
        """
        # TODO: Make sure this actually finds the area of a triangle
        v1 = frame[1] - frame[0]
        v2 = frame[2] - frame[0]
        return (1 / 2) * np.linalg.norm(np.cross(v1, v2))

    @staticmethod
    def compute_eccentricity(width, opening):
        """
        Compute the eccentricity of an ellipse
        :param width: The width of the ellipse
        :param opening: The opening of the ellipse
        :return: The eccentricity of the ellipse
        """
        e = np.zeros(width.shape)
        for k, n in enumerate(zip(width, opening)):
            w, o = abs(n[0]), abs(n[1])
            if w < o:
                e[k] = np.sqrt(1 - (w / o) ** 2)
            elif w > o:
                e[k] = np.sqrt(1 - (o / w) ** 2)
            else:
                e[k] = 1
        return e

    @staticmethod
    def three_point_difference(feature: np.ndarray, time: np.ndarray = None):
        """
        Computes three point difference for derivative
        :param feature: The feature to take the derivative of
        :param time: The times corresponding to each feature point
        :return: The time derivative of the feature
        """
        if time is None:
            h = 1
        else:
            h = np.mean(np.diff(time))

        if h == 0:
            h = 1

        dx = np.zeros(feature.shape)
        try:
            dx[0] = (1 / (2 * h)) * (-3 * feature[0] + 4 * feature[1] - feature[2])
            dx[1] = (1 / (2 * h)) * (-1 * feature[0] + feature[2])
            dx[2:] = (1 / (2 * h)) * (feature[0:-2] - 4 * feature[1:-1] + 3 * feature[2:])
        except IndexError:
            return 0
        return dx

    def get_area(self, frame):
        """
        Computes the area of the enclosed region by the landmarks
        :param frame: An (n, 2 or 3) matrix giving cartesian values for points
        :return: The area of the region
        """
        if self._dimension == 2:
            return self.poly_area_2d(frame)
        elif self._dimension == 3:
            return self.triangle_3d_area(frame)
        else:
            raise Exception(f"Area must be in 2D or 3D space, not {self._dimension}D space")

    def columns_to_numpy(self, landmark_ids: List[int], app_filter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an array where the first dimension resolves to the frame and
        the second and third dimensions holds the position data for each column
        :param landmark_ids: A list of landmark ids
        :param app_filter: Whether to apply a median filter
        :return: A tuple of numpy arrays containing the positions and rest positions
        """
        data = np.zeros((len(self._landmarks), len(landmark_ids), self._dimension))
        rest_data = np.zeros((len(self._rest_landmarks), len(landmark_ids), self._dimension))
        for i, lId in enumerate(landmark_ids):
            id_data = self._landmarks[[col.format(lId) for col in self._column_names]].to_numpy()
            id_rest_data = self._rest_landmarks[[col.format(lId) for col in self._column_names]].to_numpy()
            data[:, i, :] = id_data
            rest_data[:, i, :] = id_rest_data
        if app_filter:
            # data_size = (self._win_length, data.shape[1], 1)
            # filtered_data = ndimage.median_filter(data, size=data_size)
            # rest_data_size = (self._win_length, rest_data.shape[1], 1)
            # filtered_rest_data = ndimage.median_filter(rest_data, size=rest_data_size)

            data = self.median_filter(data)
            rest_data = self.median_filter(rest_data)
        return data, rest_data

    def median_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Applies a brute force median filter since there is a problem with the faster method
        :param data: Input positions
        :return: Median positions
        """
        # TODO: Make this memory efficient
        filtered_brute = np.zeros(shape=data.shape)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                in_data = data[:, i, j]
                filter_shape = (self._win_length,)
                brute_filtered = ndimage.median_filter(in_data, size=filter_shape)
                filtered_brute[:, i, j] = brute_filtered
        return filtered_brute

    def screen_outliers(self, feature: np.ndarray, z_threshold: float = 3) -> np.ndarray:
        """
        Finds the outliers in a feature signal
        :param feature: A 1D feature array
        :return: A mask array where True means outlier
        """
        mean = np.mean(feature)
        sd = np.std(feature)
        z_scores = (feature - mean) / sd
        return z_scores > z_threshold

    def normalize_feature(self, feature: np.ndarray, rest_feature: np.ndarray, norm_type: NormOption,
                          app_lowpass=False) -> np.ndarray:
        """
        Normalizes a feature based on the rest frame
        :param feature: A 1D feature array for distance or area of the full video
        :param norm_type: A type of normalization. Full video or running average
        :param app_lowpass: Whether to apply a low pass filter to the feature
        :return: A 1D numpy feature array of active frames normalized around the
                 values of rest frames
        """
        if norm_type == NormOption.RestAvg:
            # Then the user wants to normalize over the mean of the feature in the rest frames
            total_rest_frames = len(rest_feature)
            rest_pad = max(0, int(round((total_rest_frames - 500) / 2)))
            if rest_pad != 0:
                active_rest_feature = rest_feature[rest_pad:-rest_pad]
            else:
                active_rest_feature = rest_feature
            norm_mean = np.mean(active_rest_feature)
            if app_lowpass:
                feature = feature[~np.isnan(
                    feature)]  # TODO: Make lowpass work with nan feature without reducing the amount of frames in the feature
                feature = self.filter_feature(feature)
            normalized = feature / norm_mean  # TODO: Should I filter before or after I normalize?
            normalized = np.ma.masked_invalid(normalized, copy=False)
            return normalized
        if norm_type == NormOption.RestNDist:
            # The user wants to normalize with a normal distribution
            norm_mean = np.mean(rest_feature)
            norm_sd = np.std(rest_feature)
            return (self.get_active_feature(feature) - norm_mean) / norm_sd

    def filter_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        Apply a low pass filter to the feature
        :param feature: A 1D feature array
        :return: A 1D feature array with a lowpass filter applied
        """
        if 'Time_Stamp (s)' in self._landmarks.columns:
            return signal.filtfilt(self._fil_coeffs[0], self._fil_coeffs[1], feature)
        else:
            print("Column with name 'Time_Stamp (s)' so no low pass filter is applied")
            return feature

    @staticmethod
    def create_filter(time_stamps: np.ndarray, highcut, filter_order) -> np.ndarray:
        """"""
        fs = int(1 / (time_stamps[1] - time_stamps[0]))
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = signal.butter(filter_order, high, btype='low', output='ba')
        return np.array([b, a])

    def get_active_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        Returns the part of the feature in occurs in the active frames
        :param feature: A feature vector of the full video
        :return: A feature vector only containing the active frame values
        """
        return feature[self._active_frames]

    def eval_feature(self, feature: FeatureLandmarks, feature_type: FeatureType, app_median=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Evaluates a set of landmarks with the given feature type
        :param feature: The landmarks to take the feature over
        :param feature_type: The type of feature to extract e.g. Distance/Area
        :param app_median: Whether to apply a median filter over the positions
        :return: A tuple of arrays containing the feature values for all frames and rest frames
        """
        data, rest_data = self.columns_to_numpy(feature.value, app_filter=app_median)
        # df = pd.DataFrame({"data": data[:, 0, 0], "filtered": filtered_data[:, 0, 0], "diff": data[:, 0, 0]-filtered_data[:, 0, 0]})
        # df.to_csv("./test.csv")
        if feature_type == FeatureType.DIST:
            feature = np.linalg.norm(data[:, 0] - data[:, 1], axis=1)
            rest_feature = np.linalg.norm(rest_data[:, 0] - rest_data[:, 1], axis=1)
            outlier_mask = self.screen_outliers(rest_feature, z_threshold=2.5)
            rest_masked = np.ma.masked_array(rest_feature, mask=outlier_mask)
            return self.get_active_feature(feature), rest_masked
        if feature_type == FeatureType.AREA:
            out = np.zeros(data.shape[0])
            rest_out = np.zeros(rest_data.shape[0])
            for i, frame in enumerate(data):
                out[i] = self.get_area(frame)
            for i, frame in enumerate(rest_data):
                rest_out[i] = self.get_area(frame)
            outlier_mask = self.screen_outliers(rest_out, z_threshold=2.5)
            rest_masked = np.ma.masked_array(rest_out, mask=outlier_mask)
            return self.get_active_feature(out), rest_masked
        if feature_type == FeatureType.POS:
            return self.get_active_feature(data), rest_data

class StrokeMetrics(Metrics):
   

    def get_length_metrics(self, position) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates the Delta (max - min) and mean of a given distance between two points normalized
        by its value at rest.
        Also max and min velocity
        :return: Tuple of (Delta Distance, Mean Distance, Max Velcoity, Min Velocity)
        """
        feature, rest_feature = self.eval_feature(position, FeatureType.DIST)
        normalized = self.normalize_feature(feature, rest_feature, NormOption.RestAvg)
        velocity = self.three_point_difference(normalized)
        
        acceleration = self.three_point_difference(velocity)
        
        return normalized.max(), normalized.min(), velocity.max(), velocity.min(), acceleration.max(), acceleration.min()

    def get_area_metrics(self, positionLeft, postionRight, metric_type='CCC') -> float:
        """
        Calculates a metric between two areas
        The areas are normalized by their average values during rest
        metric_type
        CCC -> Concordance Correlation Coefficient
        dist -> average absolute difference
        pearson -> pearson correlation

        :return: metric value
        """
        left_feature, left_rest = self.eval_feature(positionLeft, FeatureType.AREA)
        right_feature, right_rest = self.eval_feature(postionRight, FeatureType.AREA)

        if metric_type == 'CCC':
            concordance = self.concordance_correlation_coefficient(left_feature, right_feature)
            return concordance

        elif metric_type == 'dist':
            # difference = np.abs(left_feature_normalized - right_feature_normalized)
            # return difference.mean()
            feature = np.abs(left_feature - right_feature)
            feature_rest = np.abs(left_rest - right_rest)
            feature_normalized = self.normalize_feature(feature, feature_rest, NormOption.RestAvg)
            return feature_normalized.mean()

        elif metric_type == 'pearson':
            pearson_corr = np.corrcoef(left_feature.reshape(1,-1), right_feature.reshape(1,-1))
            return pearson_corr[1,0]

    def get_distance_metrics(self, positionLeft, postionRight, metric_type='CCC') -> float:
        """
        Calculates a metric between two distances
        The areas are normalized by their average values during rest
        metric_type
        CCC -> Concordance Correlation Coefficient
        dist -> average absolute difference
        pearson -> pearson correlation

        :return: metric value
        """
        left_feature, left_rest = self.eval_feature(positionLeft, FeatureType.DIST)
        right_feature, right_rest = self.eval_feature(postionRight, FeatureType.DIST)

        if metric_type == 'CCC':
            concordance = self.concordance_correlation_coefficient(left_feature, right_feature)
            return concordance

        elif metric_type == 'dist':
            # difference = np.abs(left_feature_normalized - right_feature_normalized)
            # return difference.mean()
            feature = np.abs(left_feature-right_feature)
            feature_rest = np.abs(left_rest-right_rest)
            feature_normalized = self.normalize_feature(feature, feature_rest, NormOption.RestAvg)
            return feature_normalized.mean()

        elif metric_type == 'pearson':
            pearson_corr = np.corrcoef(left_feature.reshape(1, -1), right_feature.reshape(1, -1))
            return pearson_corr[1, 0]

    def compute_metrics(self, active_frames: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Computes all the metrics and returns a dataframe containing them
        :return:
        """
        metric_type = 'dist'
        if active_frames is not None:
            self._active_frames = active_frames
        all_metrics = ["O_MAX", "O_MIN", "O_MAX_VEL", "O_MIN_VEL","O_MAX_ACL","O_MIN_ACL",
                       "W_MAX", "W_MIN", "W_MAX_VEL", "W_MIN_VEL", "W_MAX_ACL","W_MIN_ACL",
                       "A_MOUTH",
                       "R_LCRC",
                       "D_0",
                       "D_1",
                       "D_2",
                       "D_3"]
                       #"D_4"]
        metrics = pd.DataFrame(columns=all_metrics)
        metrics.loc[0] = 0
        metrics.loc[0][["O_MAX", "O_MIN", "O_MAX_VEL", "O_MIN_VEL",         "O_MAX_ACL","O_MIN_ACL"]]=self.get_length_metrics(StrokeFeatureLandmarks.MouthHeight)
        metrics.loc[0][["W_MAX", "W_MIN", "W_MAX_VEL", "W_MIN_VEL", "W_MAX_ACL","W_MIN_ACL"]] = self.get_length_metrics(StrokeFeatureLandmarks.MouthWidth)
        metrics.loc[0]["A_MOUTH"] = self.get_area_metrics(StrokeFeatureLandmarks.LeftMouth,
                                                          StrokeFeatureLandmarks.RightMouth, metric_type=metric_type)
        metrics.loc[0]["R_LCRC"] = self.get_distance_metrics(StrokeFeatureLandmarks.NoseMouthL,
                                                             StrokeFeatureLandmarks.NoseMouthR, metric_type='CCC')
        metrics.loc[0]["D_0"] = self.get_distance_metrics(StrokeFeatureLandmarks.LEyebrowCanthus,
                                                          StrokeFeatureLandmarks.REyebrowCanthus, metric_type=metric_type)
        metrics.loc[0]["D_1"] = self.get_distance_metrics(StrokeFeatureLandmarks.LCanthusMouthC,
                                                          StrokeFeatureLandmarks.RCanthusMouthC, metric_type=metric_type)
        metrics.loc[0]["D_2"] = self.get_distance_metrics(StrokeFeatureLandmarks.LCanthusMouthU,
                                                          StrokeFeatureLandmarks.RCanthusMouthU, metric_type=metric_type)
        metrics.loc[0]["D_3"] = self.get_distance_metrics(StrokeFeatureLandmarks.LMouthCMouthU,
                                                          StrokeFeatureLandmarks.RMouthCMouthU, metric_type=metric_type)
        # metrics.loc[0]["D_4"] = self.get_distance_metrics(StrokeFeatureLandmarks.LMouthCMouthL, StrokeFeatureLandmarks.RMouthCMouthL, metric_type=metric_type)

        return metrics

class ClinicalMetrics(Metrics):


    def get_length_metrics(self, position) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates the Delta (max - min) and mean of a given distance between two points normalized
        by its value at rest.
        Also max and min velocity
        :return: Tuple of (Delta Distance, Mean Distance, Max Velcoity, Min Velocity)
        """
        feature, rest_feature = self.eval_feature(position, FeatureType.DIST)
        normalized = self.normalize_feature(feature, rest_feature, NormOption.RestAvg)
        velocity = self.three_point_difference(normalized)

        acceleration = self.three_point_difference(velocity)

        return normalized.max(), normalized.min(), velocity.max(), velocity.min(), acceleration.max(), acceleration.min()

    def get_area_metrics(self, positionLeft, postionRight, metric_type='CCC') -> float:
        """
        Calculates a metric between two areas
        The areas are normalized by their average values during rest
        metric_type
        CCC -> Concordance Correlation Coefficient
        dist -> average absolute difference
        pearson -> pearson correlation

        :return: metric value
        """
        left_feature, left_rest = self.eval_feature(positionLeft, FeatureType.AREA)
        right_feature, right_rest = self.eval_feature(postionRight, FeatureType.AREA)

        if metric_type == 'CCC':
            concordance = self.concordance_correlation_coefficient(left_feature, right_feature)
            return concordance

        elif metric_type == 'dist':
            # difference = np.abs(left_feature_normalized - right_feature_normalized)
            # return difference.mean()
            feature = np.abs(left_feature - right_feature)
            feature_rest = np.abs(left_rest - right_rest)
            feature_normalized = self.normalize_feature(feature, feature_rest, NormOption.RestAvg)
            return feature_normalized.mean()

        elif metric_type == 'pearson':
            pearson_corr = np.corrcoef(left_feature.reshape(1,-1), right_feature.reshape(1,-1))
            return pearson_corr[1,0]

    def get_distance_metrics(self, positionLeft, postionRight, metric_type='CCC') -> float:
        """
        Calculates a metric between two distances
        The areas are normalized by their average values during rest
        metric_type
        CCC -> Concordance Correlation Coefficient
        dist -> average absolute difference
        pearson -> pearson correlation

        :return: metric value
        """
        left_feature, left_rest = self.eval_feature(positionLeft, FeatureType.DIST)
        right_feature, right_rest = self.eval_feature(postionRight, FeatureType.DIST)

        if metric_type == 'CCC':
            concordance = self.concordance_correlation_coefficient(left_feature, right_feature)
            return concordance

        elif metric_type == 'dist':
            # difference = np.abs(left_feature_normalized - right_feature_normalized)
            # return difference.mean()
            feature = np.abs(left_feature-right_feature)
            feature_rest = np.abs(left_rest-right_rest)
            feature_normalized = self.normalize_feature(feature, feature_rest, NormOption.RestAvg)
            return feature_normalized.mean()

        elif metric_type == 'pearson':
            pearson_corr = np.corrcoef(left_feature.reshape(1, -1), right_feature.reshape(1, -1))
            return pearson_corr[1, 0]

    def compute_metrics(self, active_frames: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Computes all the metrics and returns a dataframe containing them
        :return:
        """
        metric_type = 'dist'
        if active_frames is not None:
            self._active_frames = active_frames
        all_metrics = ["O_MAX", "O_MIN", "O_MAX_VEL", "O_MIN_VEL","O_MAX_ACL","O_MIN_ACL",
                       "W_MAX", "W_MIN", "W_MAX_VEL", "W_MIN_VEL", "W_MAX_ACL","W_MIN_ACL",
                       "A_MOUTH",
                       "R_LCRC",
                       "D_0",
                       "D_1",
                       "D_2",
                       "D_3"]
                       #"D_4"]
        metrics = pd.DataFrame(columns=all_metrics)
        metrics.loc[0] = 0
        metrics.loc[0][["O_MAX", "O_MIN", "O_MAX_VEL", "O_MIN_VEL",         "O_MAX_ACL","O_MIN_ACL"]]=self.get_length_metrics(StrokeFeatureLandmarks.MouthHeight)
        metrics.loc[0][["W_MAX", "W_MIN", "W_MAX_VEL", "W_MIN_VEL", "W_MAX_ACL","W_MIN_ACL"]] = self.get_length_metrics(StrokeFeatureLandmarks.MouthWidth)
        metrics.loc[0]["A_MOUTH"] = self.get_area_metrics(StrokeFeatureLandmarks.LeftMouth,
                                                          StrokeFeatureLandmarks.RightMouth, metric_type=metric_type)
        metrics.loc[0]["R_LCRC"] = self.get_distance_metrics(StrokeFeatureLandmarks.NoseMouthL,
                                                             StrokeFeatureLandmarks.NoseMouthR, metric_type='CCC')
        metrics.loc[0]["D_0"] = self.get_distance_metrics(StrokeFeatureLandmarks.LEyebrowCanthus,
                                                          StrokeFeatureLandmarks.REyebrowCanthus, metric_type=metric_type)
        metrics.loc[0]["D_1"] = self.get_distance_metrics(StrokeFeatureLandmarks.LCanthusMouthC,
                                                          StrokeFeatureLandmarks.RCanthusMouthC, metric_type=metric_type)
        metrics.loc[0]["D_2"] = self.get_distance_metrics(StrokeFeatureLandmarks.LCanthusMouthU,
                                                          StrokeFeatureLandmarks.RCanthusMouthU, metric_type=metric_type)
        metrics.loc[0]["D_3"] = self.get_distance_metrics(StrokeFeatureLandmarks.LMouthCMouthU,
                                                          StrokeFeatureLandmarks.RMouthCMouthU, metric_type=metric_type)
        # metrics.loc[0]["D_4"] = self.get_distance_metrics(StrokeFeatureLandmarks.LMouthCMouthL, StrokeFeatureLandmarks.RMouthCMouthL, metric_type=metric_type)

        return metrics


class ALSMetrics(Metrics):
    def get_LL_path_sum(self, position) -> float:
        """
        Computes the path length of the lower lip
        :return: Distance of path
        """
        feature, rest_feature = self.eval_feature(position, FeatureType.DIST)
        normalized = self.normalize_feature(feature, rest_feature, NormOption.RestAvg)
        feature_dist = np.sum(normalized)
        return feature_dist  # TODO: Question: This varies with the length of the active period. Should it be normalized by length of video?

    def get_area_metrics(self, positionLeft, postionRight) -> [float, float, float, float]:
        """
        Calculates a metric between two areas
        The areas are normalized by their average values during rest
        metric_type
        CCC -> Concordance Correlation Coefficient
        dist -> average absolute difference
        pearson -> pearson correlation

        :return: metric value
        """
        left_area, left_rest = self.eval_feature(positionLeft, FeatureType.AREA)
        right_area, right_rest = self.eval_feature(postionRight, FeatureType.AREA)
        
        area_diff = np.abs(left_area - right_area)
        area_rest_diff = np.abs(left_rest - right_rest)
        area_diff_normalized = self.normalize_feature(area_diff, area_rest_diff, NormOption.RestAvg)
        print(area_diff_normalized)
        area_total = left_area + right_area
        area_total_rest = left_rest + right_rest
        area_total_normalized = self.normalize_feature(area_total, area_total_rest, NormOption.RestAvg)
        print(area_total_normalized)
        correlation_areas  = self.concordance_correlation_coefficient(left_area, right_area)
        print(correlation_areas)
        return area_total_normalized.mean(), np.ptp(area_total_normalized), area_diff_normalized.mean(), correlation_areas

    def get_length_metrics(self, position) -> Tuple[float, float, float, float]:
        """
        Calculates the Delta (max - min) and mean of a given distance between two points normalized
        by its value at rest.
        Also max and min velocity
        :return: Tuple of (Delta Distance, Mean Distance, Max Velcoity, Min Velocity)
        """
        feature, rest_feature = self.eval_feature(position, FeatureType.DIST)
        normalized = self.normalize_feature(feature, rest_feature, NormOption.RestAvg)
        velocity = self.three_point_difference(normalized)
        acceleration = self.three_point_difference(velocity)
        return normalized.max(), normalized.min(), velocity.max(), velocity.min(), acceleration.max(), acceleration.min()

    def get_eccentricity_metrics(self) -> Tuple[float, float]:
        """
        Computes metrics related to the eccentricity of the mouth ellipse
        :return: A tuple of (Mean Eccentricity, Range of Eccentricity)
        """
        feature_width, feature_width_rest = self.eval_feature(ALSFeatureLandmarks.MouthWidth, FeatureType.DIST)
        feature_opening, feature_opening_rest = self.eval_feature(ALSFeatureLandmarks.MouthHeight, FeatureType.DIST)
        eccentricities = self.compute_eccentricity(feature_width, feature_opening)
        return eccentricities.mean(), np.ptp(eccentricities)

    def get_distance_metrics(self, positionLeft, postionRight, metric_type='CCC') -> float:
        """
        Calculates a metric between two distances
        The areas are normalized by their average values during rest
        metric_type
        CCC -> Concordance Correlation Coefficient
        dist -> average absolute difference
        pearson -> pearson correlation

        :return: metric value
        """
        left_feature, left_rest = self.eval_feature(positionLeft, FeatureType.DIST)
        right_feature, right_rest = self.eval_feature(postionRight, FeatureType.DIST)

        if metric_type == 'CCC':
            concordance = self.concordance_correlation_coefficient(left_feature, right_feature)
            return concordance

        elif metric_type == 'dist':
            # difference = np.abs(left_feature_normalized - right_feature_normalized)
            # return difference.mean()
            feature = np.abs(left_feature-right_feature)
            feature_rest = np.abs(left_rest-right_rest)
            feature_normalized = self.normalize_feature(feature, feature_rest, NormOption.RestAvg)
            return feature_normalized.mean()

        elif metric_type == 'pearson':
            pearson_corr = np.corrcoef(left_feature.reshape(1, -1), right_feature.reshape(1, -1))
            return pearson_corr[1, 0]

    def compute_metrics(self, active_frames: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Computes all the metrics and returns a dataframe containing them
        :return:
        """
        if active_frames is not None:
            self._active_frames = active_frames
        all_metrics = ["LL_PATH",
                       "A_MEAN", "A_RANGE", "A_ABS_DIFF", "A_CCC",
                       "WIDTH_MAX", "WIDTH_MEAN", "WIDTH_VEL_MAX", "WIDTH_VEL_MIN", "WIDTH_ACL_MAX","WIDTH_ACL_MIN",
                       "HEIGHT_MAX", "HEIGHT_MIN", "HEIGHT_VEL_MAX", "HEIGHT_VEL_MIN","HEIGHT_ACL_MAX","HEIGHT_ACL_MIN",
                       "R_SYMM",
                       "E_MEAN", "E_RANGE"]
        metrics = pd.DataFrame(columns=all_metrics)
       
        
        metrics.loc[0] = 0
        metrics.loc[0]["LL_PATH"] = self.get_LL_path_sum(ALSFeatureLandmarks.LowerLip)
        metrics.loc[0][["A_MEAN", "A_RANGE", "A_ABS_DIFF", "A_CCC"]] = self.get_area_metrics(ALSFeatureLandmarks.MouthAreaLeft, ALSFeatureLandmarks.MouthAreaRight)
        metrics.loc[0][["WIDTH_MAX", "WIDTH_MEAN", "WIDTH_VEL_MAX", "WIDTH_VEL_MIN", "WIDTH_ACL_MAX","WIDTH_ACL_MIN"]] = self.get_length_metrics(ALSFeatureLandmarks.MouthWidth)
        metrics.loc[0][["HEIGHT_MAX", "HEIGHT_MIN", "HEIGHT_VEL_MAX", "HEIGHT_VEL_MIN","HEIGHT_ACL_MAX","HEIGHT_ACL_MIN"]] = self.get_length_metrics(ALSFeatureLandmarks.MouthHeight)
        metrics.loc[0]["R_SYMM"] = self.get_distance_metrics(ALSFeatureLandmarks.NoseMouthL, ALSFeatureLandmarks.NoseMouthR)
        metrics.loc[0][["E_MEAN", "E_RANGE"]] = self.get_eccentricity_metrics()

        return metrics



class ALSorStrokeMetrics(StrokeMetrics, ALSMetrics):
    def compute_metrics(self, active_frames: Optional[Iterable[int]] = None) -> pd.DataFrame:
        stroke_metrics = StrokeMetrics.compute_metrics(self, active_frames)
        als_metrics = ALSMetrics.compute_metrics(self, active_frames)
        all_metrics = pd.concat([stroke_metrics, als_metrics], axis=1)
        return all_metrics


if __name__ == "__main__":
    df = pd.read_csv("/Users/aidandempster/projects/uhn/vidProc/demo/preprocessed.csv")
    test_metrics = StrokeMetrics(df, rest_frames=range(10, 100), active_frames=range(300, 600))
    metrics = test_metrics.compute_metrics()
    print(metrics)
