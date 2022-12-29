# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:07:21 2022

@author: Lenovo
"""
import pandas as pd
import numpy as np
import torch

class Change_Knowledge_Variable():
    def __init__(self,input_):
        '''
        input_type:dataframe
        input_size:[d,k]
             d:时间序列长度
             k:知识变量个数
        知识变量含义：
            potVolt:槽电压
            filterResist:滤波电阻
            smoothResist:平滑电阻
            slopeData:斜率
            sumSlopeData:累斜
            fluctDelta:针振
            wavingDelta:摆动
            settingVoltMax:设定电压最大值
            settingVoltMin:设定电压最小值
            anodeChangeToNow:换极距今
        '''
        self.input_=input_
        self.output_=pd.DataFrame({'feature0':0,'feature1':0},index=[0])
        
    def change_knowledge_variable(self):
        self.output_['feature0']=np.where(self.input_['fluctDelta'].max()>15,0.8,0.2)#针振是否超过15
        #self.output_['feature1']=np.where(self.input_['fluctDelta'].max()>15,0.1,0.8)#针振是否超过15
        #self.output_['feature0']=self.input_['fluctDelta'].max()#针振最大值
       # self.output_['feature3']=np.where(self.input_['wavingDelta'].max()>6,1,0)#摆动最大值
        self.output_['feature1']=self.input_['wavingDelta'].max()#摆动最大值
        #self.output_['feature2']=self.input_['sumSlopeData'].var()
        self.output_['feature2']=self.input_['sumSlopeData'].var()
        self.output_['feature3']=self.input_['slopeData'].var()
        #self.output_['feature2']=self.input_['potVolt'].max()/self.input_['potVolt'].var()
        return self.output_
        
    