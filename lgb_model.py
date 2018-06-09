# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import logging.handlers
from data_preprocess import *
import getFeat as getfeat
"""Train the lightGBM model"""

LOG_FILE = 'log/lgb_train.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def lgb_fit(config, X_train, y_train,X_valid,y_valid,columns_col):

    params = config.params
    max_round = config.max_round
    early_stop_round = config.early_stop_round
    save_model_path = config.save_model_path

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    watchlist = [dtrain,dvalid]
    best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds = early_stop_round)
    best_round = best_model.best_iteration
    best_logloss = best_model.best_score
    print("==============================\n",best_logloss)

    best_model.save_model(save_model_path)
   
    #-feature_importance
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] =columns_col
    dfFeature['score'] = best_model.feature_importance()
    dfFeature = dfFeature.sort("score",ascending=False)
    dfFeature.to_csv('lgb_feature_score.csv',index=False)

    return best_model, best_logloss, best_round


def lgb_predict(model, test_data,test_instance_id):
    from sklearn.metrics import log_loss
    y_pred_prob = model.predict(test_data)
    print(y_pred_prob)
    test_instance_id['predicted_score'] = y_pred_prob
    test_index_org = pd.read_csv('data/round2_ijcai_18_test_b_20180510.txt',sep = " ")[['instance_id']]
    res_test = pd.merge(test_index_org,test_instance_id,on='instance_id',how = 'left')
    now = time.strftime("%m%d__%H_%M")
    res_test.to_csv('result/result_lgb_{}.txt'.format(now), index=False,sep = " ")


class Config(object):
    def __init__(self,max_round):
        self.params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':  {'binary_logloss'},
          'min_child_weight': 1.5,
          'num_leaves': 2**5,
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.5,
          'colsample_bylevel': 0.5,
          'learning_rate': 0.05,
          'seed': 2018,
          'nthread': 32,
          'silent': True,
        }

        self.max_round = max_round
        self.early_stop_round = 50
        self.save_model_path = 'model/lgb.txt'


#---线下验证----
def lgbTrain_off_line(trainData, valData,drop_columns_list):
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
    lgb_model, best_logloss, best_round = lgb_fit(config, train_data.values, y_train.values,val_data.values,y_val.values,columns_col)
  
   
    result_message = 'best_round: {}, best_logloss: {}'.format(best_round,best_logloss)
    logger.info(result_message)
    print(result_message)

    return best_round
    

 
def lgbTrain(trainData, valData, testData,beat_round,drop_columns_list):
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
    lgb_model, best_logloss, best_round = lgb_fit(config, train_data.values, y_train.values,val_data.values,y_val.values,columns_col)
    
    # predict
    lgb_predict(lgb_model, test_data.values,test_instance_id)
   
        
   

def lgb_trian_final():
    #--选择线上还是线下，1：线上 0：线下
    on_off_flag = 1
    drop_columns_list = [
                        # 'context_id_lasttime_delta',
                        # 'user_context_id_lasttime_delta',
                        # 'user_context_id_trade_prep_count',
                        # 'user_id_context_id_browse',
                        # 'context_id_nexttime_delta',
                        # 'user_context_id_nexttime_delta',
                        # 'user_id_item_id_click',
                        # 'user_id_context_id_hour_browse',
                        # 'user_id_context_id_day_browse',
                        # 'user_shop_id_trade_prep_count',
                        # 'user_brand_id_trade_prep_count',
                        # 'user_category_id_trade_prep_count',
                        # 'item_category_1',
                        # 'context_id_trade_prep_count',
                        # 'user_id_context_id_click',
                        # 'tmp_count',
                        # 'day',
                        # 'user_id_context_id_rate',
                        # 'user_item_id_trade_prep_count',
                        # 'user_id_day_hour_context_id_click',
                        # 'user_id_day_context_id_click',
                        # 'user_id_item_id_day_click',
                        # 'user_id_item_id_hour_browse',
                        # 'context_id_click',
                        # 'user_id_item_id_hour_click',
                        # 'user_id_shop_id_hour_click',
                        # 'user_id_item_brand_id_hour_click',
                        # 'user_id_item_city_id_hour_click',
                        # 'user_id_context_page_id_hour_click',
                        # 'user_id_context_id_hour_click',
                        # 'user_id_item_category_hour_click',
                        # 'user_id_context_id_day_rate',
                        # 'user_id_shop_id_day_rate',
                        # 'user_id_item_category_day_click',
                        # 'user_id_shop_id_hour_rate',
                        # 'user_id_context_id_day_click',
                        # 'user_id_context_page_id_day_click',
                        # 'user_id_item_city_id_day_click',
                        # 'user_id_item_city_id_hour_rate',
                        # 'user_id_shop_id_day_click',
                        # 'user_id_context_id_hour_rate',
                        # 'user_city_id_trade_prep_count'

                        ]
    #=====================训练======================
    train_together_data,train_data,val_data,test_data = load_feat()
    if on_off_flag == 0:
        lgbTrain_off_line(train_data, val_data,drop_columns_list)
    else:
      beat_round = lgbTrain_off_line(train_data, val_data,drop_columns_list)
      lgbTrain(train_together_data, val_data, test_data,beat_round,drop_columns_list)


    
   
   














