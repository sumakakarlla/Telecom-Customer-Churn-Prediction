'''
In this file we are going to take 4 numerical columns and we are going to select the
best columns using feature selection techniques
1.constant technique
2.quasi constant technique
3.hypothesis testing
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from log_code import setup_logging
logger = setup_logging("FS")
from sklearn.feature_selection import VarianceThreshold
constant_reg = VarianceThreshold(threshold=0.0)
quasi_reg = VarianceThreshold(threshold=0.1)
from scipy.stats import pearsonr

def feature_sel(X_train_numerical, X_test_numerical,y_train):
    try:
        logger.info(f"Before Train data shape and columns : {X_train_numerical.shape} : {X_train_numerical.columns}")
        logger.info(f"Before Test data shape and columns: {X_test_numerical.shape} : {X_test_numerical.columns}")
        '''
        Implementing Constant Technique 
        '''
        constant_reg.fit(X_train_numerical)
        logger.info(f"Number of columns that  are good: {sum(constant_reg.get_support())}")
        logger.info(f"Number of columns that are Bad : {sum(~constant_reg.get_support())}")
        logger.info(f"Columns that should be  removed : {X_train_numerical.columns[~constant_reg.get_support()]}")
        '''
        using  constant technique no columns r removed because all are good 
        '''
        '''
        implementing quasi  technique 
        '''
        quasi_reg.fit(X_train_numerical)
        logger.info(f"Number of columns that are good: {sum(quasi_reg.get_support())}")
        logger.info(f"Number of columns that are Bad : {sum(~quasi_reg.get_support())}")
        logger.info(f"Columns that should be removed : {X_train_numerical.columns[~quasi_reg.get_support()]}")

        '''
        using  constant technique no columns r removed because all are good 
        '''
        '''
        implementing hypothesis testing 
        '''

        values = []
        for i in X_train_numerical.columns:
          values.append(pearsonr(X_train_numerical[i] , y_train))
        logger.info(values)
        values = np.array(values)
        p_values = values[: , 1]
        s = pd.Series(p_values,index = X_train_numerical.columns)
        plt.figure(figsize=(5,3))
        s.plot.bar()
        plt.show()
        X_train_numerical = X_train_numerical.drop(['SeniorCitizen'], axis=1)
        X_test_numerical = X_test_numerical.drop(['SeniorCitizen'], axis=1)
        logger.info(f"After Feature Selection : {X_train_numerical.shape} : \n : {X_train_numerical.columns}")
        logger.info(f"After Feature Selection : {X_test_numerical.shape} : \n : {X_test_numerical.columns}")
        return X_train_numerical, X_test_numerical

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")
        '''
        From hypothesis testing Senior citizen column is removed
        now we have 3 numerical columns
        '''
