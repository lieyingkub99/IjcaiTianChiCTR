# -*- coding:utf-8 -*-
from getFeat import *
from laodData import *
from lgb_model import *
from other_feat import *
#from xgb_model import *

if __name__ == '__main__':
	#print("loadData...\n")
	#data_split()
	
	print("other_feat...\n")
	other_feat_handle()
	
	print("getFeat...\n")
	feat_handle()
	
	print("lgb_model...\n")
	lgb_trian_final()

	#print("xgb_model...")
	#xgb_trian_final()

	#print("lgb_model...\n")
	#lgb_trian_final()
