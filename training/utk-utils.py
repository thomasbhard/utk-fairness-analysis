import os
import glob

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from PIL import Image

dataset_folder_name = 'dataset/UTKFace'
outputfile_name = "df_predictions_all_test.csv"


TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198

TRAIN_WITH_WEIGTHS = True

dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())