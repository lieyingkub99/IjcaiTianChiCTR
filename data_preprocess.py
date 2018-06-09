#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import time
from bayes_smoothing import *
from sklearn.preprocessing import LabelEncoder
import copy



def roll_browse_fetch(df, column_list):
    print("==========================roll_browse_fetch ing==============================")
    df = df.sort('context_timestamp')
    df['tmp_count'] = df['status']
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        df['%s_browse' %c] = df.groupby(pair)['tmp_count'].cumsum()
    del df['tmp_count']
    return df

def roll_click_fetch(df, column_list):
    print("==========================roll_click_fetch ing==============================")
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        df['%s_click' %c] = df.groupby(pair)['is_trade'].cumsum()
        df['%s_click' %c] = df['%s_click' %c]-df['is_trade']
    return df

def roll_rate_fetch(df, column_list):
    df = roll_browse_fetch(df,column_list)
    df = roll_click_fetch(df,column_list)
    print("==========================roll_rate_fetch ing==============================\n")
    for c in column_list:
        if isinstance(c, (list, tuple)):
            c = '_'.join(c)
        df['%s_rate' %c] = bs_utilize(df['%s_browse' %c], df['%s_click' %c])
        # del df['%s_browse' %c]
    return df

#===================================按天的转化率==============================
def roll_browse_day(df, column_list):
    df = df.sort('context_timestamp')
    df['tmp_count'] = df['status']
    df_data_temp =df.copy() 
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        pair.append('day')
        df_temp = df.groupby(pair)['tmp_count'].agg({"browse_temp":np.sum}).reset_index()
        pair_temp =copy.copy(pair)
        pair_temp.remove('day')
        df_temp["{}_day_browse".format(c)] = df_temp.groupby(pair_temp)["browse_temp"].cumsum()
        df_temp["{}_day_browse".format(c)] = df_temp["{}_day_browse".format(c)] - df_temp['browse_temp']
        del df_temp['browse_temp']
        df_data_temp = pd.merge(df_data_temp,df_temp,how = "left",on = pair )
    del df['tmp_count']
    return df_data_temp

def roll_click_day_hour(df,column_list):
    df = df.sort('context_timestamp')
    df_data_temp =df.copy() 
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        pair.append('day')
        df_temp = df.groupby(pair)['is_trade'].agg({"click_temp":np.sum}).reset_index()
        pair_temp = copy.copy(pair)
        pair_temp.remove('day')
        df_temp["{}_day_click".format(c)] = df_temp.groupby(pair_temp)["click_temp"].cumsum()
        df_temp["{}_day_click".format(c)] = df_temp["{}_day_click".format(c)] - df_temp['click_temp']
        del df_temp['click_temp']
        df_data_temp = pd.merge(df_data_temp,df_temp,how = "left",on = pair)
    return df_data_temp

def roll_rate_day(df,column_list):
    print("==========================roll_rate_day ing==============================")
    df = roll_browse_day(df,column_list)
    df =roll_click_day(df,column_list)
    for c in column_list:
        if isinstance(c, (list, tuple)):
            c = '_'.join(c)
        df['%s_day_rate' %c] = bs_utilize(df['%s_day_browse' %c], df['%s_day_click' %c])
        # del df['%s_day_browse'%c]
        # del df['%s_day_click'%c]
    return df
#===================================按天小时的转化率==============================
def roll_browse_day_hour(df, column_list):
    df = df.sort('context_timestamp')
    df['tmp_count'] = df['status']
    df_data_temp =df.copy() 
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        pair.append('day')
        pair.append('hour')
        df_temp = df.groupby(pair)['tmp_count'].agg({"browse_temp":np.sum}).reset_index()
        pair_temp =copy.copy(pair)
        pair_temp.remove('day')
        pair_temp.remove('hour')
        df_temp["{}_day_hour_browse".format(c)] = df_temp.groupby(pair_temp)["browse_temp"].cumsum()
        df_temp["{}_day_hour_browse".format(c)] = df_temp["{}_day_hour_browse".format(c)] - df_temp['browse_temp']
        del df_temp['browse_temp']
        df_data_temp = pd.merge(df_data_temp,df_temp,how = "left",on = pair )
    del df['tmp_count']
    return df_data_temp
def roll_click_day_hour(df,column_list):
    df = df.sort('context_timestamp')
    df_data_temp =df.copy() 
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        pair.append('day')
        pair.append('hour')
        df_temp = df.groupby(pair)['is_trade'].agg({"click_temp":np.sum}).reset_index()
        pair_temp = copy.copy(pair)
        pair_temp.remove('day')
        pair_temp.remove('hour')
        df_temp["{}_day_hour_click".format(c)] = df_temp.groupby(pair_temp)["click_temp"].cumsum()
        df_temp["{}_day_hour_click".format(c)] = df_temp["{}_day_hour_click".format(c)] - df_temp['click_temp']
        del df_temp['click_temp']
        df_data_temp = pd.merge(df_data_temp,df_temp,how = "left",on = pair)
    return df_data_temp

def roll_rate_day_hour(df,column_list):
    print("==========================roll_rate_day ing==============================")
    df = roll_browse_day_hour(df,column_list)
    df =roll_click_day_hour(df,column_list)
    for c in column_list:
        if isinstance(c, (list, tuple)):
            c = '_'.join(c)
        df['%s_day_hour_rate' %c] = bs_utilize(df['%s_day_hour_browse' %c], df['%s_day_hour_click' %c])
        # del df['%s_day_browse'%c]
        # del df['%s_day_click'%c]
    return df



#===================================按小时的转化率==============================
def roll_browse_hour(df, column_list):
    df = df.sort('context_timestamp')
    df['tmp_count'] = df['status']
    df_data_temp =df.copy() 
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        pair.append('hour')
        df_temp = df.groupby(pair)['tmp_count'].agg({"browse_temp":np.sum}).reset_index()
        pair_temp =copy.copy(pair)
        pair_temp.remove('hour')
        df_temp["{}_hour_browse".format(c)] = df_temp.groupby(pair_temp)["browse_temp"].cumsum()
        df_temp["{}_hour_browse".format(c)] = df_temp["{}_hour_browse".format(c)] - df_temp['browse_temp']
        del df_temp['browse_temp']
        df_data_temp = pd.merge(df_data_temp,df_temp,how = "left",on = pair )
    del df['tmp_count']
    return df_data_temp
def roll_click_hour(df,column_list):
    df = df.sort('context_timestamp')
    df_data_temp =df.copy() 
    for c in column_list:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        pair.append('hour')
        df_temp = df.groupby(pair)['is_trade'].agg({"click_temp":np.sum}).reset_index()
        pair_temp = copy.copy(pair)
        pair_temp.remove('hour')
        df_temp["{}_hour_click".format(c)] = df_temp.groupby(pair_temp)["click_temp"].cumsum()
        df_temp["{}_hour_click".format(c)] = df_temp["{}_hour_click".format(c)] - df_temp['click_temp']
        del df_temp['click_temp']
        df_data_temp = pd.merge(df_data_temp,df_temp,how = "left",on = pair)
    return df_data_temp

def roll_rate_hour(df,column_list):
    print("==========================roll_rate_hour ing==============================")
    df = roll_browse_hour(df,column_list)
    df =roll_click_hour(df,column_list)
    for c in column_list:
        if isinstance(c, (list, tuple)):
            c = '_'.join(c)
        df['%s_hour_rate' %c] = bs_utilize(df['%s_hour_browse' %c], df['%s_hour_click' %c])
       
    return df



def label_encoding(df, columns):
    for c in columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
    return df
# # #----------------统计特征-----------------
# def get_last_diff_statistic(data,col_list, n_last_diff):
#     print("=======get_last_diff============\n")
#     data_temp = data
#     col_id = col_list[0],col_list[1]
#     data = data.sort_values([col_id, 'timestamp'])
#     data['next_id'] = data[col_id].shift(-1)
#     data['next_actionTime'] = data.timestamp.shift(-1)
#     data = data.loc[data.next_id == data[col_id]].copy()
#     data['action_diff'] = data['next_actionTime'] - data['timestamp']
#     if n_last_diff is not None:
#         df_n_last_diff = data.groupby(col_id, as_index=False).tail(n_last_diff).copy()
#         df_last_diff_statistic = df_n_last_diff.groupby(col_id, as_index=False).action_diff.agg({
#             '{}_last_{}_action_diff_mean'.format(col_id,n_last_diff): np.mean,
#             '{}_last_{}_action_diff_std'.format(col_id,n_last_diff): np.std,
#             '{}_last_{}_action_diff_max'.format(col_id,n_last_diff): np.max,
#             '{}_last_{}_action_diff_min'.format(col_id,n_last_diff): np.min
#         })
#     else:
#         grouped_user = data.groupby(col_id, as_index=False)
#         n_last_diff = 'all'
#         df_last_diff_statistic = grouped_user.action_diff.agg({
#             '{}_last_{}_action_diff_mean'.format(col_id,n_last_diff): np.mean,
#             '{}_last_{}_action_diff_std'.format(col_id,n_last_diff): np.std,
#             '{}_last_{}_action_diff_max'.format(col_id,n_last_diff): np.max,
#             '{}_last_{}_action_diff_min'.format(col_id,n_last_diff): np.min
            
#         })
#     res_data = pd.merge(data_temp,df_last_diff_statistic,how="left",on = col_id)

#     return res_data
# #-----------------------时间特征-----------------------
# # #--时间间隔特征、
# def chafen(df):
#     return pd.DataFrame(np.diff(df,axis = 0))

# def get_last_diff(data, col_list,n_last_diff):
#     """获取最后 n_last_diff 个动作之间的时间间隔"""
#     print("=======get_last_diff============\n")
#     for col in col_list:
#         col_sort = col.copy()
#         col_sort.append('timestamp')
#         data = data.sort_values(col_sort,ascending = False)
#         data_temp = data.groupby(col)['timestamp'].apply(chafen).reset_index()
#         data_temp.columns = [col[0],col[1],'level','time_gap']
#         data_temp = data_temp.loc[data_temp.level<n_last_diff]
#         data_temp['time_gap'] = -1*data_temp['time_gap']
#         data_temp['level'] = str(col[0])+"_"+str(col[1])+"_last_time_gap"+ data_temp['level'].astype('str')
#         data_temp = pd.pivot_table(data_temp,index=[col[0],col[1]],values='time_gap',columns='level').reset_index()
#         res_data = pd.merge(data,data_temp,how="left",on = [col[0],col[1]])
#     return res_data



#--时间间隔特征
def time_diff_feat(data,col_list):
    print("get tiem diff...")
    for col in col_list:
        col_sort = copy.copy(col)
        col_sort.append('timestamp')
        data_temp = data.sort(col_sort,ascending = True)
        data_temp['{}_{}_time_diff'.format(col[0],col[1])] = data_temp.groupby(col)['timestamp'].apply(lambda x:x.diff())
        data['{}_{}_time_diff'.format(col[0],col[1])] = data_temp['{}_{}_time_diff'.format(col[0],col[1])].fillna(0)
    return data 


def CombinationFeature(data):
    print("==============convert_data===============")
    
    data['tm_hour'] = data['hour'] + data['min']/60
    data['tm_hour_sin'] = data['tm_hour'].map(lambda x:np.sin((x-12)/24*2*np.pi))
    data['tm_hour_cos'] = data['tm_hour'].map(lambda x:np.cos((x-12)/24*2*np.pi))
    data_time=data[['user_id','day','hour','min']]
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})    
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})   
    user_query_day_hour_min = data.groupby(['user_id', 'day', 'hour','min']).size().reset_index().rename(columns={0: 'user_query_day_hour_min'})
    user_query_day_hour_min_sec = data.groupby(['user_id', 'day', 'hour','min','sec']).size().reset_index().rename(columns={0: 'user_query_day_hour_min_sec'})
    user_day_hourmin_mean= data_time.groupby(['user_id', 'day']).mean().reset_index().rename(columns={'hour': 'mean_hour','min':'mean_minuite'})  
    user_day_hourmin_std= data_time.groupby(['user_id', 'day']).std().reset_index().rename(columns={'hour': 'std_hour','min':'std_minuite'}) 
    user_day_hourmin_max= data_time.groupby(['user_id', 'day']).max().reset_index().rename(columns={'hour': 'max_hour','min':'max_minuite'})
    user_day_hourmin_min= data_time.groupby(['user_id', 'day']).min().reset_index().rename(columns={'hour': 'min_hour','min':'min_minuite'})   
 
    #-------merge-----
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    data = pd.merge(data, user_query_day_hour, 'left',on=['user_id', 'day', 'hour'])
    data = pd.merge(data, user_query_day_hour_min, 'left',on=['user_id', 'day', 'hour','min'])
    data = pd.merge(data, user_query_day_hour_min_sec, 'left',on=['user_id', 'day', 'hour','min','sec'])
    data = pd.merge(data, user_day_hourmin_mean, 'left',on=['user_id','day'])
    data = pd.merge(data, user_day_hourmin_std, 'left',on=['user_id','day'])
    data = pd.merge(data, user_day_hourmin_max, 'left',on=['user_id','day'])
    data = pd.merge(data, user_day_hourmin_min, 'left',on=['user_id','day'])
   
   
   #==============================click_feat================================
    data_temp = data.copy()
    columns_click = [
                     #--单个--
                     ['user_id'],
                     ['item_id'],
                     ['context_id'],
                     ['shop_id'],
                     ['item_brand_id'],
                     ['item_city_id'],
                     ['context_page_id'],
                     ['item_category_2'],
                     ['item_property_list_1'],
                     ['item_property_list_2'],
                     ['item_property_list_3'],
                     ['predict_category_property_A'],
                     ['predict_category_property_B'],
                     ['predict_category_property_C'],
                     ['predict_category_property_A_1'],
                     ['predict_category_property_B_1'],
                     ['predict_category_property_C_1'],
                     ['predict_category_property_A_2'],
                     ['predict_category_property_B_2'],
                     ['predict_category_property_C_2'],

                     #--加day--
                     ['user_id','day','item_id'],
                     ['user_id','day','context_id'],
                     ['user_id','day','shop_id'],
                     ['user_id','day','item_brand_id'],
                     ['user_id','day','item_city_id'],
                     ['user_id','day','context_page_id'],
                     ['user_id','day','item_category_2'],
                     ['user_id','day','item_property_list_1'],
                     ['user_id','day','item_property_list_2'],
                     ['user_id','day','item_property_list_3'],
                     ['user_id','day','predict_category_property_A'],
                     ['user_id','day','predict_category_property_B'],
                     ['user_id','day','predict_category_property_C'],
                     ['user_id','day','predict_category_property_A_1'],
                     ['user_id','day','predict_category_property_B_1'],
                     ['user_id','day','predict_category_property_C_1'],
                     ['user_id','day','predict_category_property_A_2'],
                     ['user_id','day','predict_category_property_B_2'],
                     ['user_id','day','predict_category_property_C_2'],
                     
                     #--加day,hour--
                     ['user_id','day','hour','item_id'],
                     ['user_id','day','hour','context_id'],
                     ['user_id','day','hour','shop_id'],
                     ['user_id','day','hour','item_brand_id'],
                     ['user_id','day','hour','item_city_id'],
                     ['user_id','day','hour','context_page_id'],
                     ['user_id','day','hour','item_category_2'],
                     ['user_id','day','hour','item_property_list_1'],
                     ['user_id','day','hour','item_property_list_2'],
                     ['user_id','day','hour','item_property_list_3'],
                     ['user_id','day','hour','predict_category_property_A'],
                     ['user_id','day','hour','predict_category_property_B'],
                     ['user_id','day','hour','predict_category_property_C'],
                     ['user_id','day','hour','predict_category_property_A_1'],
                     ['user_id','day','hour','predict_category_property_B_1'],
                     ['user_id','day','hour','predict_category_property_C_1'],
                     ['user_id','day','hour','predict_category_property_A_2'],
                     ['user_id','day','hour','predict_category_property_B_2'],
                     ['user_id','day','hour','predict_category_property_C_2'],

                     #--加hour--
                     ['user_id','hour','item_id'],
                     ['user_id','hour','shop_id'],
                     ['user_id','hour','item_brand_id'],
                     ['user_id','hour','item_city_id'],
                     ['user_id','hour','context_page_id'],
                     ['user_id','hour','item_category_2'],
                     ['user_id','hour','item_property_list_1'],
                     ['user_id','hour','item_property_list_2'],
                     ['user_id','hour','item_property_list_3'],
                     ['user_id','hour','predict_category_property_A'],
                     ['user_id','hour','predict_category_property_B'],
                     ['user_id','hour','predict_category_property_C'],
                     ['user_id','hour','predict_category_property_A_1'],
                     ['user_id','hour','predict_category_property_B_1'],
                     ['user_id','hour','predict_category_property_C_1'],
                     ['user_id','hour','predict_category_property_A_2'],
                     ['user_id','hour','predict_category_property_B_2'],
                     ['user_id','hour','predict_category_property_C_2']
                    ]

    for c in columns_click:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        click_temp = data_temp.groupby(pair).size().reset_index().rename(columns={0: '{}_click'.format(c)}) 
        data = pd.merge(data, click_temp, how = 'left',on=pair)

    return data
    
def load_feat():
   
    train_together_data = pd.read_csv("data_handle/train_together_data_after.csv")
    train_together_data = train_together_data[train_together_data['context_timestamp'] > '2018-09-06 23:59:59']

    val_data = train_together_data[train_together_data['context_timestamp'] > '2018-09-07 10:59:59']
    train_data = train_together_data[train_together_data['context_timestamp'] <= '2018-09-07 10:59:59']
   
    # train_data = pd.read_csv("data_handle/train_data.csv")     
    # val_data = pd.read_csv("data_handle/valid_data.csv")
    test_data = pd.read_csv("data_handle/test_data_after.csv")

    del train_together_data['context_timestamp']
    del train_data['context_timestamp']
    del val_data['context_timestamp']
    del test_data['context_timestamp']
  
    return train_together_data,train_data,val_data,test_data

if __name__ == '__main__':
    pass
