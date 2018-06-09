#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import time
from data_preprocess import *


def time2cov(time_):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_))


def delete_column(data_tuple, column_tuple):
    for i in data_tuple:
        for j in column_tuple:
            del i[j]
    if len(data_tuple) == 1:
        data_tuple = data_tuple[0]
    return data_tuple


def train_valid_seperate(train_together):
    train_copy = train_together.copy()
    valid = train_copy[train_copy['context_timestamp']>'2018-09-07 02:59:59']
    train = train_copy[train_copy['context_timestamp']<='2018-09-07 02:59:59']
    
    return train, valid


def data_all_seperate(data_all):
    train = data_all[data_all['status_split'] == 0].copy()
    valid = data_all[data_all['status_split'] == 1].copy()
    test = data_all[data_all['status_split'] == 2].copy()
    train_together = data_all[(data_all['status_split'] == 0) | (data_all['status_split'] == 1)].copy()
    train, valid, test, train_together = delete_column((train, valid, test, train_together), ['status_split'])
    return train, valid, test, train_together


# ===================数据预处理==================
def pre_process(data):
    print('pre_process...')
    # ==================item_category_list===================
    for i in range(3):
        data['item_category_%d' % (i + 1)] = data['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else np.nan
        )
    data["item_category"] = data["item_category_list"].apply(lambda x: x.split(";"))
    data['item_category'] = data['item_category'].astype(str)
    del data['item_category_list']

    # ===============item_property_list==================
    for i in range(3):
        data['item_property_list_%d' % (i + 1)] = data['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else np.nan
        )
    del data['item_property_list']
    # ===================predict_category_property===================
    # -先按照；分开
    ABC = ['A', 'B', 'C']
    for i in range(3):
        data['predict_category_property_front_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['predict_category_property']
    # -按照：分开取第一个
    for i in range(3):
        data['predict_category_property_{}'.format(ABC[i])] = data['predict_category_property_front_%d' % (i)].apply(
            lambda x: str(x).split(":")[0] if len(str(x).split(":")) > 1 else " "
        )
    # -按照：分开取第一个
    for i in range(3):
        data['predict_category_property_back_%d' % (i)] = data['predict_category_property_front_%d' % (i)].apply(
            lambda x: str(x).split(":")[1] if len(str(x).split(":")) > 1 else " "
        )
        del data['predict_category_property_front_%d' % (i)]
        for j in range(3):
            data['predict_category_property_{}_{}'.format(ABC[i], j + 1)] = data[
                'predict_category_property_back_%d' % (i)].apply(
                lambda x: str(x).split(",")[j] if len(str(x).split(",")) > j else " "
            )
        del data['predict_category_property_back_%d' % (i)]

    data = data.replace(" ", np.nan)
    # ================context_timestamp=====================
    data['timestamp'] = data['context_timestamp']
    data['context_timestamp'] = data['context_timestamp'].values.astype(np.int64)
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)

    data['day'] = data['context_timestamp'].apply(lambda x: int(x[8:10]))
    data['hour'] = data['context_timestamp'].apply(lambda x: int(x[11:13]) )
    data['min'] = data['context_timestamp'].apply(lambda x: int(x[14:16]))
    data['sec'] = data['context_timestamp'].apply(lambda x: int(x[17:19]))



    # ===================简单组合特征=====================
    data['sale_price'] = (data['item_sales_level'] + 1) * (data['item_price_level'] + 1)
    data['sale_collect'] = (data['item_sales_level'] + 1) * (data['item_collected_level'] + 1)
    data['price_collect'] = (data['item_price_level'] + 1) * (data['item_collected_level'] + 1)
    return data


# ===============转化率特征==================
def conversion_feat(data_all):
    columns_list = [
        'user_id',
        'item_id',
        'shop_id',
        'item_city_id',
        'context_page_id',
        'item_brand_id',
        'predict_category_property_A',
        #'predict_category_property_B',
        #'predict_category_property_C',
        'item_property_list_1',
        #'item_property_list_2',
        #'item_property_list_3',
        # 'item_sales_level',
        # 'item_collected_level',
        # 'item_price_level',
        # 'item_pv_level',
        ['user_id', 'item_id'],
        ['user_id', 'shop_id'],
        ['user_id', 'item_brand_id'],
        ['user_id', 'item_city_id'],
        ['user_id', 'context_page_id'],
        ['user_id', 'context_id'],
        ['user_id', 'item_category'],
        ['user_id', 'item_property_list_1'],
        #['user_id', 'item_property_list_2'],
        #['user_id', 'item_property_list_3'],
        # ['user_id','item_sales_level'],
        # ['user_id','item_collected_level'],
        # ['user_id','item_price_level'],
        # ['user_id','item_pv_level'],
        ['user_id','predict_category_property_A']
        #['user_id','predict_category_property_B'],
        #['user_id','predict_category_property_C'],

    ]
    # ---计算当前时间之前的转化率----
    data_all = roll_rate_fetch(data_all, columns_list)
    # ---按天进行转化----
    data_all = roll_rate_day(data_all, columns_list)
    # ---按天,小时进行转化----
    data_all = roll_rate_day_hour(data_all, columns_list)
    #----按小时转化---------
    data_all = roll_rate_hour(data_all, columns_list)
       
    return data_all


def feat_handle():
    print('read  data...')
    train = pd.read_csv('data/round2_train.txt', sep=" ")
    testa = pd.read_csv('data/round2_ijcai_18_test_a_20180425.txt', sep=" ")
    testb =  pd.read_csv('data/round2_ijcai_18_test_b_20180510.txt', sep=" ")
    test = pd.concat([testa,testb],axis = 0)
    print("train_test_testA_testB_shape:  ", train.shape, test.shape)
    train['is_trade'] = train['is_trade'].astype('int64')
    train = train.drop_duplicates(['instance_id'])  # 把instance id去重

    train = pre_process(train)
    test = pre_process(test)
    test['status'] = 0
    train['status'] = 1
    test['is_trade'] = 0

    train, valid = train_valid_seperate(train)
    train['status_split'] = 0
    valid['status_split'] = 1
    test['status_split'] = 2

    data_all = pd.concat([train, valid, test], ignore_index=True).reset_index(0, drop=True)
    col_list = [
        ['user_id', 'predict_category_property_A'],
        ['user_id', 'predict_category_property_B'],
        ['user_id', 'predict_category_property_C'],
        #['user_id', 'item_property_list_1'],
        #['user_id', 'item_property_list_2'],
        #['user_id', 'item_property_list_3']
    ]
    data_all = time_diff_feat(data_all, col_list)
    data_all = CombinationFeature(data_all)  # 一系列组合特征
    data_all = conversion_feat(data_all)  # 转化率特征

    CATEGORY_COLUMNS = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'item_price_level',
                        'item_sales_level', 'item_collected_level', 'item_pv_level', 'shop_review_num_level',
                        'shop_star_level']
    data_all = label_encoding(data_all, CATEGORY_COLUMNS)

    a_gap_time=pd.read_csv('data_handle/gap_time.csv')
    a_trade_prep_count=pd.read_csv('data_handle/trade_prep_count.csv')
    data_all =pd.merge(data_all,a_gap_time,how = 'left',on=['instance_id','status'])
    data_all =pd.merge(data_all,a_trade_prep_count,how = 'left',on=['instance_id','status'])


    train, valid, test, train_together = data_all_seperate(data_all)
    train, valid, test, train_together = delete_column((train, valid, test, train_together,),
                                                       [
                                                           # 'context_timestamp',
                                                           'timestamp',
                                                           'status',
                                                           'item_category',
                                                       ])

    print("train_together:  ", train_together.shape)
    print("train:  ", train.shape)
    print("valid:  ", valid.shape)
    print("test:  ", test.shape)

    train_together.to_csv("data_handle/train_together_data_after.csv", index=False)
    # train.to_csv("data_handle/train_data.csv", index=False)
    # valid.to_csv("data_handle/valid_data.csv", index=False)
    test.to_csv("data_handle/test_data_after.csv", index=False)

    return train_together, train, valid, test


# if __name__ == '__main__':
#     feat_handle()
