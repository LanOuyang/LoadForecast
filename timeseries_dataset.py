import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import torch
from torch.utils.data import Dataset
import torchvision

import os.path

from data_utils import *


class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 path, 
                 weather_path,
                 input_len, 
                 output_len, 
                 output_size = 1,
                 stride = 1,
                 lag = 1,
                 target_column = 'WHE',
                 scaler = None,
                 weather_encoder = None,
                 datetime_encoder = None,
                 mode='train',
                 row_limit=None):
        
        self.df = pd.read_csv(path, index_col=False)
        self.weather = pd.read_csv(weather_path, index_col=False)

        self.input_len = input_len
        self.output_len = output_len
        self.output_size = output_size
        self.stride = stride
        self.lag = lag

        self.target_column = target_column

        self.weather_encoder = weather_encoder
        self.datetime_encoder = datetime_encoder
        self.num_scaler = scaler

        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        train, val, test = np.split(self.df, [
            int(.6 * len(self.df)), int(.8 * len(self.df))
        ])
        if mode == "train":
            self.df = train
        elif mode == "val":
            self.df = val
        elif mode == "test":
            self.df = test

        # Use normalized target or unnormalized target?
        # self.target = self.df[self.target_column].to_numpy()
        self.preprocess()
        self.features = self.df.to_numpy()
        self.target = self.df[self.target_column].to_numpy()

        self.inseqs, self.outseqs = generateSequencesFromTimeSeries(self.features, 
                                                                    self.target, 
                                                                    self.input_len, 
                                                                    self.output_len,
                                                                    self.output_size,
                                                                    self.stride,
                                                                    self.lag)

    def __len__(self):
        return len(self.inseqs)

    def __getitem__(self, index):

        assert not np.any(np.isnan(self.inseqs[index]))

        return self.inseqs[index], self.outseqs[index]
    
    def preprocess(self):
        """
        power data:
        1. Convert timestamp

        weather data:
        1. Encode categorical features
        2. Decompose 'weather' feature
        
        Join two tables,
        encode datetime features, 
        fill missing values (from weather dataset), 
        and normalize numerical features
        """
        df = convertTimestampToDatetime(self.df)
        df.drop(columns='MHE',inplace=True)

        drop_features = ['Date/Time', 'Visibility Flag', 'Hmdx Flag', 'Wind Chill Flag'] # + 'Time'
        
        cat_features = ['Data Quality', 'Temp Flag', 'Dew Point Temp Flag', 'Rel Hum Flag', 'Wind Dir (10s deg)', 
                        'Wind Dir Flag', 'Wind Spd Flag', 'Stn Press Flag', 'Weather']
        
        num_features = ['Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 
                        'Visibility (km)', 'Stn Press (kPa)', 'Hmdx', 'Wind Chill']
        
        self.weather.drop(columns=drop_features, inplace=True)

        weather, self.weather_encoder = encodeCategoricalFeatures(self.weather, cat_features, self.weather_encoder)
        weather = decomposeWeather(weather)

        weather['Hour'] = weather['Time'].apply(lambda t: t.split(':')[0])

        joinon_fields = ['Year', 'Month', 'Day', 'Hour']
        weather[joinon_fields] = weather[joinon_fields].astype('int16')

        df = pd.merge(df, weather, on=joinon_fields, how='inner')

        df.drop(columns=['Time', 'Year'], inplace=True)

        df, self.datetime_encoder = encodeCategoricalFeatures(df, ['Month', 'Day', 'Week', 'Weekday', 'Hour'], self.datetime_encoder)

        e_num_features = ['WHE', 'RSE', 'EQE', 'OFE', 'UTE', 'FRE', 'HTE', 'HPE', 
                          'UNE', 'TVE', 'B2E', 'BME', 'GRE', 'FGE', 'EBE', 
                          'DNE', 'B1E', 'CWE', 'DWE', 'WOE', 'CDE', 'OUE']
        self.df, self.num_scaler = scaleNumericalFeatures(df, num_features+e_num_features, self.num_scaler)

        self.df.fillna(0, inplace=True)

        # print(self.df.shape)
        # print(self.df.columns)

