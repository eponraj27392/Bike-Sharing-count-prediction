# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:41:46 2020

@author: eponr
"""

"""
Import req libraries 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from tqdm import tqdm
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import Pool, CatBoostRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import model_selection 

import skopt
from functools import partial
from skopt import space
from skopt import gp_minimize



"""
Filepath /directory
"""
FILE_PATH         = "C:\\Users\\eponr\\Desktop\\Open_Source_project\\2. Bike Sharing dataset"
HOUR_DF_NAME      = 'hour.csv'
DAY_DATA_NAME     = 'day.csv' 
SAVE_PATH         = 'C:\\Users\\eponr\\Desktop\\Open_Source_project\\2. Bike Sharing dataset\\model'



"""
Variables
"""
CAT_FEATURES      = ['season', 'yr', 'mnth', 'holiday', 'weekday','workingday', 'weathersit']
WEATHER_FEATURES  = ['temp', 'atemp', 'hum', 'windspeed']
TARGET            = ['cnt']
UNWANTED_FEATURES = ['instant', 'dteday', 'casual', 'registered', 'temp']



"""
Possible Model development Areas with TIme series
1. First 19 days of data  & predict remaining days cnt in a month
2. Simple : Just predict the last 50 days of predicted cnt
3. Predict the next 15 days cnt

"""


class preprocessing:
    def __init__(self, df_hour, df_day, cat_features = None, unwant_features = None, weather_features = None,  
                 target_features = None, 
                 scaler          = None,
                 outlier_removal = None,
                 OHE             = None,
                 FREQ_ENCODER    = None,
                 FACT_ENCODER    = None,
                 STANDARDIZATION = None, 
                 test_value      = None):
        
        self.df_hour             =   df_hour
        self.df_day              =   df_day
        self.cat_features        =   cat_features
        self.unwant_features     =   unwant_features
        self.weather_features    =   weather_features
        self.train_features      =   self.cat_features + self.weather_features[1:] 
        self.target_features     =   target_features
        self.outlier_removal     =   outlier_removal
        self.scaler              =   scaler
        self.df                  =   self.df_day
        self.test_value          =   test_value

        self.OHE                 =   OHE
        self.freq_encoder        =   FREQ_ENCODER
        self.fact_encoder        =   FACT_ENCODER
        self.standardize         =   STANDARDIZATION
        self.RealScaler          =   None

        
    def df_outlier(self):
        
        # Remove outlier
        if self.outlier_removal:
            # cnt = 22 indx = 667 = storm
            self.df                 =  self.df.drop(667).reset_index(drop= True)
            # hum = 0 idx =  68 = 
            self.df                 =  self.df.drop(68).reset_index(drop= True)
        
        else:    
            cnt_ls_100                  = self.df[self.df.cnt < 100].index
            respective_mnth             = self.df[(self.df.yr == self.df_day.loc[cnt_ls_100]['yr'].values[0]) & 
                                                      (self.df.mnth == self.df.loc[cnt_ls_100]['mnth'].values[0])]
            respective_mnth_cnt_mean    = self.df.loc[[i for i in respective_mnth.cnt.index if i != cnt_ls_100]]['cnt'].mean()
            self.df.loc[cnt_ls_100]     = self.df.loc[cnt_ls_100].assign(cnt = respective_mnth_cnt_mean)
            
            # humidity zero in 10.3.2011, this is not possible suddenly, previous day has 77 % and the next day has 60 % humidiy
            self.df['hum']              = self.df['hum'].apply(lambda x : np.nan if x == 0 else x)
            self.df['hum']              = self.df['hum'].interpolate()

        return self.df
    
    def df_ConvertCatFeatures(self):
        for feature in self.cat_features:
             self.df[feature] = self.df[feature].astype('category') 
        return self.df
    
    
    def df_encoder(self):
        # Implement OHE using get_dummies
        season      =  pd.get_dummies(self.df['season'],prefix='season',drop_first=False)
        year        =  pd.get_dummies(self.df['yr'],prefix='yr',drop_first=False)
        month       =  pd.get_dummies(self.df['mnth'],prefix='mnth',drop_first=False)
        holiday     =  pd.get_dummies(self.df['holiday'],prefix='hlday',drop_first=False)
        weekday     =  pd.get_dummies(self.df['weekday'],prefix='wkday',drop_first=False)
        workingday  =  pd.get_dummies(self.df['workingday'],prefix='wrkday',drop_first=False)
        weathersit  =  pd.get_dummies(self.df['weathersit'],prefix='weather',drop_first=False)
        
        self.df     =  pd.concat([self.df, season, year, month, holiday, weekday, workingday, weathersit], axis = 1)
        
        train_col   = self.df.loc[:, 'season_1' :].columns.tolist()  + self.weather_features[1:] +  ['cnt']
        self.df     = self.df[train_col]
        
        trainX      =  self.df.iloc[:-self.test_value,  : -1]
        testX       =  self.df.iloc[ -self.test_value :, :-1]
        trainY      =  self.df.iloc[:-self.test_value, -1:]
        testY       =  self.df.iloc[ -self.test_value:, -1:]
        return self.df, trainX, testX, trainY, testY
    
    
    def df_freqencoder(self):
        for c in tqdm(self.cat_features):
            self.df[c+'_freq'] = self.df[c].map(self.df.groupby(c).size() / self.df.shape[0])
            self.df[c+'_freq'] = self.df[c+'_freq'].astype('float')
        
        train_col   =  self.df.loc[:, 'season_freq' :].columns.tolist()  + self.weather_features[1:] +  ['cnt']
        self.df     =  self.df[train_col]
       
        trainX      =  self.df.iloc[:-self.test_value,  : -1]
        testX       =  self.df.iloc[ -self.test_value :, :-1]
        trainY      =  self.df.iloc[:-self.test_value, -1:]
        testY       =  self.df.iloc[ -self.test_value:, -1:]
        
        
        return self.df, trainX, testX, trainY, testY
    

    def df_factencoder(self):
        for c in tqdm(self.cat_features):
            indexer = pd.factorize(self.df[c], sort=True)[1]
            self.df[c] = indexer.get_indexer(self.df[c])
            
        train_col   =  self.cat_features + self.weather_features[1:] +  ['cnt']
        self.df     =  self.df[train_col]
        
        trainX      =  self.df.iloc[:-self.test_value,  : -1]
        testX       =  self.df.iloc[ -self.test_value :, :-1]
        trainY      =  self.df.iloc[:-self.test_value, -1:]
        testY       =  self.df.iloc[ -self.test_value:, -1:]   
        return self.df, trainX, testX, trainY, testY
   
    def Scaling(self):
        self.RealScaler =   self.scaler.fit(self.df['cnt'].values.reshape(-1,1))
        self.df['cnt']  =   self.RealScaler.transform(self.df['cnt'].values.reshape(-1,1))
        return  self.RealScaler, self.df
    

    def TrainTestSplit(self):
        
        # call outlier fn
        self.df     =  self.df_outlier()
        
        # convert_cat
        self.df     =  self.df_ConvertCatFeatures()
        
        # scaler
        if self.standardize:
            self.RealScaler, self.df     =  self.Scaling()
        
        # dummy
        if self.OHE:
            self.df, trainX, testX, trainY, testY  =  self.df_encoder()
            
        elif self.freq_encoder:
            self.df, trainX, testX, trainY, testY  =  self.df_freqencoder()
            
        elif self.fact_encoder:
            self.df, trainX, testX, trainY, testY  =  self.df_factencoder()
            
        else:
            trainX      =   self.df.iloc[:-self.test_value, :][self.train_features]
            testX       =   self.df.iloc[-self.test_value :, :][self.train_features]
            trainY      =   self.df.iloc[:-self.test_value, -1:]
            testY       =   self.df.iloc[-self.test_value:, -1:]

        return self.RealScaler, trainX, testX, trainY, testY
    
    
    def evaluation_metrics(self,  testY, pred):
        
        if self.standardize:
            testY   =  pd.DataFrame(self.RealScaler.inverse_transform(testY.values.reshape(-1,1)))
            pred    =  self.RealScaler.inverse_transform(pred.reshape(-1,1))
        
        print("R2: %.2f" %metrics.r2_score(testY, pred))
        print('MAE:  %.2f',metrics.mean_absolute_error(testY, pred))
        print('MSE:  %.2f',metrics.mean_squared_error(testY, pred))
        print('RMSE: %.2f',np.sqrt(metrics.mean_squared_error(testY, pred)))
        return testY, pred
    
    def LinearRegression(self, trainX, testX, trainY, testY):
        lr_reg                       = LinearRegression().fit(trainX.values, trainY.values)
        pred = lr_reg.predict(testX)
        
        testY, pred = self.evaluation_metrics(testY, pred)
        self.df_plot(testY, pred, 'Linear Regression')
        return lr_reg, pred
    
    def KnnRegression(self, trainX, testX, trainY, testY):
        knn_reg = KNeighborsRegressor()
        knn_reg.fit(trainX.values, trainY.values)
        pred = knn_reg.predict(testX)
        
        testY, pred = self.evaluation_metrics(testY, pred)
        self.df_plot(testY, pred, 'KNN Regression')
        return knn_reg, pred
    
    def CatBoost(self, trainX, testX, trainY, testY):
        
        train_pool = Pool(trainX,  trainY,   cat_features=None)
        test_pool  = Pool(testX,  testY,   cat_features=None)

        cat_reg = CatBoostRegressor() # iterations=500,  depth=4,   learning_rate=1,  loss_function='RMSE'
        cat_reg.fit(train_pool, verbose = 100)
        pred = cat_reg.predict(test_pool)
            
        testY, pred = self.evaluation_metrics(testY, pred)
        self.df_plot(testY, pred, 'CatBoost Regression')
        return cat_reg, pred
        
    def RandomForest(self, trainX, testX, trainY, testY):
        # grid search params
        params_based_gridsearch_cv = {'criterion'  : 'mse', 'max_depth' : 13, 'n_estimators' : 250}
        
        rfc_reg = RandomForestRegressor(**params_based_gridsearch_cv) # random_state = 12548, 
        rfc_reg.fit(trainX, trainY.values)
        pred = rfc_reg.predict(testX)
        
        testY, pred = self.evaluation_metrics(testY, pred)
        self.df_plot(testY, pred, 'RandomForest Regression')
        return rfc_reg, pred
    
    def XgbRegressor(self, trainX, testX, trainY, testY):
        
        # since XGB dont handle cat features 
        if (self.OHE == False) & (self.freq_encoder == False) & (self.fact_encoder == False):
            for feature in ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']:
                trainX[feature] = trainX[feature].astype('int64')
                testX[feature]  = testX[feature].astype('int64')
        
        
        # grid earch params        
        params_based_gridsearch_cv = {'min_child_weight'  : 1.0, 'gamma' : 0.3, 'subsample' : 0.6, 
                                      'colsample_bytree' :1.0, 'max_depth' : 4,  'n_estimators' : 350}
        xgb_reg = xgb.XGBRegressor(**params_based_gridsearch_cv)
        xgb_reg.fit(trainX.values, trainY.values)
        pred = xgb_reg.predict(testX.values)
        
        testY, pred = self.evaluation_metrics(testY.values, pred)
        self.df_plot(testY, pred, 'XG Boost Regression')
        return xgb_reg, pred


    def df_plot(self, testY, pred, Model_name = None):
        
        plt.figure()
        plt.plot(testY, color = 'blue', label = 'Actual')
        plt.plot(pred, color = 'red', label = 'pred')
        plt.legend(['Actual', 'Predicted'])
        plt.title(str(Model_name), fontdict =  {'fontsize' : 24})
        plt.show()
        
        return 
    
    def save(self, save_path, model, model_name = None):
        with open(save_path + '/' + model_name,'wb') as f:
            pickle.dump(model,f)
            return

if __name__ == '__main__' :
    df_hour = pd.read_csv(FILE_PATH + '/' + HOUR_DF_NAME) 
    df_day  = pd.read_csv(FILE_PATH + '/' + DAY_DATA_NAME) 
    
    """
    call the class 
    """
    cls_preprocess = preprocessing(df_hour, 
                                   df_day, 
                                   cat_features     = CAT_FEATURES,
                                   unwant_features  = UNWANTED_FEATURES,
                                   weather_features = WEATHER_FEATURES,
                                   target_features  = TARGET, 
                                   scaler           = MinMaxScaler(), # MinMaxScaler() StandardScaler
                                   outlier_removal  = True,
                                   
                                   OHE              = False,
                                   FREQ_ENCODER     = False,
                                   FACT_ENCODER     = True,
                                   STANDARDIZATION  = False,
                                   test_value       = 146  # 20 % of test data
                                   )
    
    
    scaling_method, trainX, testX, trainY, testY = cls_preprocess.TrainTestSplit()
    
    #"""
    # print('Linear Regression :')
    # lr_reg, lr_pred  = cls_preprocess.LinearRegression(trainX, testX, trainY, testY)
    # print()
    # print('KNN Regression :')
    # knn_reg, knn_pred = cls_preprocess.KnnRegression(trainX, testX, trainY, testY)
    # print()
    # print('CatBoost Regressor :')
    # cat_reg, cat_pred = cls_preprocess.CatBoost(trainX, testX, trainY, testY)
    # print()
    # print('Random Forest Regressor :')
    # rfc_reg, rfc_pred = cls_preprocess.RandomForest(trainX, testX, trainY, testY)
    # print()
    print('XG Boost Regressor :')
    xgb_reg, xgb_pred   = cls_preprocess.XgbRegressor(trainX, testX, trainY, testY)
    #"""
 
    save_model = cls_preprocess.save(SAVE_PATH, xgb_reg, 'XGB_linear_model_with_r2Value_0.77.pkl')

 
#https://github.com/anujvyas/IPL-First-Innings-Score-Prediction-Deployment


# scaling the o/p or use log
# implement NN
# learn cat boost/XG boost 
# create flask & 
# 
    



# # Load the Random Forest CLassifier model

MODEL_PATH = 'C:\\Users\\eponr\\Desktop\\Open_Source_project\\2. Bike Sharing dataset\\model'
MODEL_NAME = 'XGB_linear_model_with_r2Value_0.77.pkl'
regressor = pickle.load(open(MODEL_PATH + '/' +  MODEL_NAME, 'rb'))

a = [0,1,11,0,2,1,1,0.52,0.80,0.35]
arr = np.array([a])
arr.shape


int(regressor.predict(arr)[0])



#      testX.shape
     
     
# xgb_reg.predict(testX.values)     
     