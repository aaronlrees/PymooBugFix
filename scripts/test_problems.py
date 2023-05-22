import numpy as np

from sklearn.preprocessing import StandardScaler


class Scalarise():

    def __init__(self, arr):
        self.scaler = StandardScaler()
        # self.arr_scaled = np.float32(self.scaler.fit_transform(arr))
        self.arr_scaled = self.scaler.fit_transform(arr)
        self.std_dev = np.std(arr)

    def scale(self, new_arr):
        #  return np.float32(self.scaler.fit_transform(new_arr))
        return self.scaler.fit_transform(new_arr)


def rescale(scaled_res, scaler):
    return scaler.inverse_transform(scaled_res.reshape(-1, 1))
