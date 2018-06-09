import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from data_preprocess import *

def time2cov(time_):
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))


def print_data(data):
	print(data.shape)
	data['timestamp'] = data['context_timestamp']
	data['context_timestamp'] = data['context_timestamp'].apply(time2cov)
	# data = data.loc[data.context_timestamp>'2018-09-03 23:59:59']
	data['context_timestamp'] = data['timestamp'] 
	del data['timestamp'] 

	print(data.shape)
	data.to_csv("data/train_data_new.txt",index=False,sep = " ")

def data_split():
	print('read  data...')
	train = pd.read_csv('data/round2_train.txt', sep=" ")
	print("train_data..")
	print_data(train)



