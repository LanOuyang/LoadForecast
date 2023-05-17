import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from datetime import datetime
import pytz


def convertTimestampToDatetime(df, 
                               ts_col='UNIX_TS', 
                               tz = pytz.timezone('America/Vancouver'),
                               dt_format = '%Y %m %d %W %w %H',
                               dt_fields=['Year', 'Month', 'Day', 'Week', 'Weekday', 'Hour'], 
                               dt_type='int16'):

    dt = df[ts_col].apply(lambda ts: datetime.fromtimestamp(ts, tz=tz).strftime(dt_format).split())

    # Can not cast 'object' to 'int16'
    # dt = pd.DataFrame(dt.to_list(), columns=dt_fields, dtype=dt_type)

    # dt = pd.DataFrame(dt.to_list(), columns=dt_fields)
    # dt = dt.astype(dt_type, copy=False)

    # problem is here
    # df_withDatetime = pd.concat([df.drop(columns=ts_col),dt], axis=1)
    # print(df_withDatetime.isna().all())

    df.drop(columns=ts_col, inplace=True)
    df[dt_fields] = dt.to_list()
    df[dt_fields] = df[dt_fields].astype(dt_type)

    return df

def encodeCategoricalFeatures(df, 
                      cat_feature_cols,
                      encoder = None):

    if encoder != None:
        encoded_features = encoder.transform(df[cat_feature_cols].to_numpy())
    
    else:
        encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[cat_feature_cols].to_numpy())
    
    # OHE should treat na as its own category
    # print(encoder.get_feature_names_out(cat_feature_cols))
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_feature_cols))

    return pd.concat([df.drop(columns=cat_feature_cols), encoded_df], axis=1), encoder

def scaleNumericalFeatures(df,
                           num_feature_cols,
                           scaler = None):
    if scaler:
        #print(df[num_feature_cols].to_numpy()[0])
        scaled_features = scaler.transform(df[num_feature_cols].to_numpy())
        #print(scaled_features[0])

    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[num_feature_cols].to_numpy())

    scaled_df = pd.DataFrame(scaled_features, columns=num_feature_cols)

    return pd.concat([df.drop(columns=num_feature_cols), scaled_df], axis=1), scaler

def decomposeWeather(df):
    # An example: 'Weather_Freezing Drizzle - Fog'
    
    cols = df.columns
    drop_cols = []

    # print(df.shape)

    for col in cols:
        if " - " in col:
            # print(col)

            idx = df[col] == 1

            drop_cols.append(col)

            col = col.rstrip('\n')
            col = col.lstrip('Weather_')

            labels = col.split(' - ')

            for label in labels:
                if 'Weather_' + label not in cols:
                    df['Weather_' + label] = 0

                df.loc[idx, 'Weather_'+label] = 1

    df = df.drop(columns = drop_cols)

    # print(df.shape)

    return df

def generateSequencesFromTimeSeries(input,
                                    target, 
                                    input_len,
                                    output_len,
                                    output_size = 1,
                                    stride = 1,
                                    lag = 1):
    
    assert len(input) == len(target), 'incompatible size of input and target'
    assert lag > 0

    seq_num = len(input) - lag - output_len - output_size + 2

    x = [input[i:i + input_len] for i in range(0, seq_num, stride)]

    y = [[target[i + j + lag : i + j + lag + output_size] for j in range(output_len)] for i in range(0, seq_num, stride)]
         
    return x, np.array(y)

