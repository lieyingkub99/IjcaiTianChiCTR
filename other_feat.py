#-*-coding:utf-8-*-
import pandas as pd
import time

import numpy as np
from datetime import datetime,timedelta
import datetime


def other_feat_handle():
    ##处理类目类特征，将类目类特征分为一列一列
    data_train = pd.read_csv('data/round2_train.txt',sep=" ")
    testa = pd.read_csv('data/round2_ijcai_18_test_a_20180425.txt', sep=" ")
    testb =  pd.read_csv('data/round2_ijcai_18_test_b_20180510.txt', sep=" ")
    data_test = pd.concat([testa,testb],axis = 0)

    # data_train = pd.read_csv('data/data_trian_test.txt', sep=" ")
    # data_test = pd.read_csv('data/data_trian_test.txt', sep=" ")
    # testb =  pd.read_csv('data/round1_ijcai_18_test_b_20180418.txt', sep=" ")
    # test = pd.concat([testa,testb],axis = 0)

    # data_testb=pd.read_csv('data/round1_ijcai_18_test_b_20180418.txt',delim_whitespace=True)
    # data_test = pd.concat([data_testa,data_testb],axis = 0)


    data_test['status'] = 0
    data_test['is_trade'] = 0
    data_train=data_train.drop_duplicates(['instance_id'])

    data_train['status'] = 1
    train_data = pd.concat([data_train,data_test],ignore_index=True)

    train_data["datetime"] = train_data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    train_data["day"] = train_data["datetime"].apply(lambda x: x.day)
    train_data["hour"] = train_data["datetime"].apply(lambda x: x.hour)

    ##统计各类别在此次出现前的count数
    def count_cat_prep(df,column,newcolumn):
        count_dict = {}
        df[newcolumn] = 0
        data = df[[column,newcolumn]].values
        for cat_list in data:
            if cat_list[0] not in count_dict:
                count_dict[cat_list[0]] = 0
                cat_list[1] = 0
            else:
                count_dict[cat_list[0]] += 1
                cat_list[1] = count_dict[cat_list[0]]
        df[[column,newcolumn]] = data

    train_data['user_item_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_id'].astype(str)
    train_data['user_shop_id'] = train_data['user_id'].astype(str)+"_"+train_data['shop_id'].astype(str)
    train_data['user_brand_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_brand_id'].astype(str)
    train_data['item_category'] = train_data['item_category_list'].astype(str)
    train_data['user_category_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_category'].astype(str)
    train_data['user_context_id'] = train_data['user_id'].astype(str)+"_"+train_data['context_id'].astype(str)
    train_data['user_city_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_city_id'].astype(str)
    ##统计各类别在总样本中的count数
    for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
        count_cat_prep(train_data,column,column+'_click_count_prep')
        
    for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
        data_temp = train_data.groupby([column])[column].agg({'{}_count'.format(column): np.size}).reset_index()
        train_data = pd.merge(train_data, data_temp, how="left", on=column)
        # train_data = train_data.join(train_data[column].value_counts(),on = column ,rsuffix = '_count')
        
    ##前一次或后一次点击与现在的时间差（trick）
    def lasttime_delta(column):    
        train_data[column+'_lasttime_delta'] = 0
        data = train_data[['context_timestamp',column,column+'_lasttime_delta']].values
        lasttime_dict = {}
        for df_list in data:
            if df_list[1] not in lasttime_dict:
                df_list[2] = -1
                lasttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
                lasttime_dict[df_list[1]] = df_list[0]
        train_data[['context_timestamp',column,column+'_lasttime_delta']] = data

    def nexttime_delta(column):    
        train_data[column+'_nexttime_delta'] = 0
        data = train_data[['context_timestamp',column,column+'_nexttime_delta']].values
        nexttime_dict = {}
        for df_list in data:
            if df_list[1] not in nexttime_dict:
                df_list[2] = -1
                nexttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
                nexttime_dict[df_list[1]] = df_list[0]
        train_data[['context_timestamp',column,column+'_nexttime_delta']]= data

    for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
        lasttime_delta(column)
        
    train_data = train_data.sort('context_timestamp',ascending=False)

    for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
        nexttime_delta(column)
        
    train_data = train_data.sort('context_timestamp')

    ##统计各类别在此次出现前的 trade转化数
    def trade_prep_count(column):
        train_data[column+'_trade_prep_count'] = 0
        for day in train_data.day.unique():
            if day == 18:
                train_data.loc[train_data.day==day,column+'_trade_prep_count'] = -1
            else:
                trade_dict = train_data[train_data.day<day].groupby(column)['is_trade'].sum().to_dict()
                train_data.loc[train_data.day==day,column+'_trade_prep_count'] = train_data.loc[train_data.day==day,column].apply(lambda x: trade_dict[x] if x in trade_dict else 0)

    for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
        trade_prep_count(column)


    a_prep = train_data[['instance_id','user_id_click_count_prep','item_id_click_count_prep',
                         'item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep',
                         'user_shop_id_click_count_prep','user_brand_id_click_count_prep',
                         'user_category_id_click_count_prep','context_id_click_count_prep',
                         'item_city_id_click_count_prep','user_context_id_click_count_prep',
                         'user_city_id_click_count_prep','status']]
    a_count = train_data[['instance_id','user_id_count','item_id_count','item_brand_id_count','shop_id_count',
                          'user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
                          'context_id_count','item_city_id_count','user_context_id_count','user_city_id_count','status']]

    a_instance=train_data['instance_id']
    a_gap_time=train_data[['user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta',
                            'shop_id_lasttime_delta','user_item_id_lasttime_delta','user_shop_id_lasttime_delta',
                            'user_brand_id_lasttime_delta','user_category_id_lasttime_delta','context_id_lasttime_delta',
                            'item_city_id_lasttime_delta','user_context_id_lasttime_delta','user_city_id_lasttime_delta',
                            'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta',
                            'shop_id_nexttime_delta','user_item_id_nexttime_delta','user_shop_id_nexttime_delta',
                            'user_brand_id_nexttime_delta','user_category_id_nexttime_delta','context_id_nexttime_delta',
                            'item_city_id_nexttime_delta','user_context_id_nexttime_delta','user_city_id_nexttime_delta',
                            'status']]
    a_trade_prep_count=train_data[['user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count',
                                   'shop_id_trade_prep_count','user_item_id_trade_prep_count','user_shop_id_trade_prep_count',
                                   'user_brand_id_trade_prep_count','user_category_id_trade_prep_count',
                                   'context_id_trade_prep_count','item_city_id_trade_prep_count','user_context_id_trade_prep_count',
                                   'user_city_id_trade_prep_count','status']]

    a_instance = pd.DataFrame(a_instance)
    a_gap_time = pd.DataFrame(a_gap_time)
    a_gap_time = pd.concat([a_instance,a_gap_time],axis=1)
    a_trade_prep_count = pd.concat([a_instance,a_trade_prep_count],axis=1)
    a_prep=a_prep.sort_index()
    a_count=a_count.sort_index()
    a_gap_time=a_gap_time.sort_index()
    a_trade_prep_count = a_trade_prep_count.sort_index()


    a_gap_time.to_csv('data_handle/gap_time.csv',index = False)
    a_trade_prep_count.to_csv('data_handle/trade_prep_count.csv',index=False)

