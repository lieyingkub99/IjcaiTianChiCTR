# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
import logging.handlers
from data_preprocess import *

"""Train the lightGBM model."""

LOG_FILE = 'log/xgb_train.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def xgb_fit(config, X_train, y_train,X_valid,y_valid,columns_col):

    params = config.params
    max_round = config.max_round
    early_stop_round = config.early_stop_round
    save_model_path = config.save_model_path
    #------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    best_model = xgb.train(params, dtrain, max_round, evals=watchlist, early_stopping_rounds=early_stop_round)
    best_round = best_model.best_iteration
    best_logloss = best_model.best_score
    best_model.save_model(save_model_path)

    #----特征重要性排名-----
    feature_score = best_model.get_fscore()
    print(len(feature_score))
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))

    with open('xgb_feature_score.csv', 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



    return best_model, best_logloss, best_round


def xgb_predict(model, test_data,test_instance_id):
    from sklearn.metrics import log_loss
    test_data = xgb.DMatrix(test_data)
    y_pred_prob = model.predict(test_data)
    
    data_test = pd.DataFrame()
    data_test["instance_id"] = test_instance_id
    data_test["predicted_score"] = y_pred_prob


    test_index_org = pd.read_csv('data/round2_ijcai_18_test_b_20180510.txt',sep = " ")[['instance_id']]
    res_test = pd.merge(test_index_org,data_test,on='instance_id',how = 'left')
    now = time.strftime("%m%d__%H_%M")
    res_test.to_csv('result/result_xgb_{}.txt'.format(now), index=False, sep=" ")


class Config(object):
    def __init__(self,max_round):
        self.params = {
            'learning_rate': 0.05,
            'objective':'binary:logistic',
            'eval_metric': "logloss",
            'max_depth': 6,
            'min_child_weight': 1.5,
            'gamma': 0,
            'lambda':10,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'colsample_bylevel':0.5,
            'seed':2018,
            'silent': True,
            'nthread':32,
            # 'scale_pos_weight':1
        }
        self.max_round = max_round
        self.early_stop_round = 50
        self.save_model_path = 'model/xgb.txt'


#---线下验证----
def xgbTrain_off_line(trainData, valData,drop_columns_list):
    train_data = trainData.copy()
    val_data  = valData.copy()

    train_data = train_data.drop(drop_columns_list,axis=1)  
    val_data = val_data.drop(drop_columns_list,axis=1) 

    y_train = train_data.pop("is_trade")
    y_val = val_data.pop("is_trade")
    data_message = 'train_data.shape={}, X_test.shape={}'.format(train_data.shape, val_data.shape)
    print(data_message)
    logger.info(data_message)
    columns_col = list(train_data.columns)
    config = Config(2000)     
    val_data = val_data[columns_col]
    lgb_model, best_logloss, best_round = xgb_fit(config, train_data, y_train,val_data,y_val,columns_col)
  
   
    result_message = 'best_round: {}, best_logloss: {}'.format(best_round,best_logloss)
    logger.info(result_message)
    print(result_message)

    return best_round
    

 
def xgbTrain(trainData, valData, testData,beat_round,drop_columns_list):
    train_data = trainData.copy()
    val_data  = valData.copy()
    test_data = testData.copy()
    test_instance_id = test_data[["instance_id"]]

    train_data = train_data.drop(drop_columns_list,axis=1)  
    test_data = test_data.drop(drop_columns_list,axis=1)  
    val_data = val_data.drop(drop_columns_list,axis=1) 
   
    y_train = train_data.pop("is_trade")
    y_val = val_data.pop("is_trade")
   
    columns_col = list(train_data.columns)
    print("train-shape,valid-shape",train_data.shape, val_data.shape)
    config = Config(beat_round)      # train model
    val_data = val_data[columns_col]
    test_data = test_data[columns_col]
    lgb_model, best_logloss, best_round = xgb_fit(config, train_data, y_train,val_data,y_val,columns_col)
    
    # predict
    xgb_predict(lgb_model, test_data,test_instance_id)
   
        
   
def xgb_trian_final():

    #--选择线上还是线下，1：线上 0：线下
    on_off_flag = 1 
    drop_columns_list = [
                        
                        ]
    #=====================训练======================
    train_together_data,train_data,val_data,test_data = load_feat()

    if on_off_flag == 0:
        xgbTrain_off_line(train_data, val_data,drop_columns_list)
    else:
        beat_round = xgbTrain_off_line(train_data, val_data,drop_columns_list)
        xgbTrain(train_together_data, val_data, test_data,beat_round,drop_columns_list)


   



