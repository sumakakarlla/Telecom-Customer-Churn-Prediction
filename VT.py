import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from log_code import setup_logging
logger = setup_logging("VT")
from scipy.stats import yeojohnson

def var_transformation(X_train_numerical , X_test_numerical):
    try:
        logger.info(f"Train Data : {X_train_numerical.shape} : \n : {X_train_numerical.columns}")
        logger.info(f"Test Data : {X_test_numerical.shape} : \n {X_test_numerical.columns}")

        '''
        As senior Citizen column is having binary values we r not applying transformation
        as tenure and MonthlyCharges columns skewness is close to 0 we r not applying transformation
        noe we r applying  yeojohnson for TotalCharges and save it  in new column ,then we will  delete old column
        '''

        X_train_numerical['TotalCharges_yeo'],lam_value = yeojohnson(X_train_numerical['TotalCharges'])
        X_test_numerical["TotalCharges_yeo"] ,lam_value = yeojohnson(X_test_numerical['TotalCharges'])

        X_train_numerical = X_train_numerical.drop(['TotalCharges'],axis=1)
        X_test_numerical = X_test_numerical.drop(['TotalCharges'],axis=1)

        logger.info(f"After Train Data : {X_train_numerical.shape} : \n : {X_train_numerical.columns}")
        logger.info(f"After Test Data : {X_test_numerical.shape} : \n {X_test_numerical.columns}")

        return X_train_numerical, X_test_numerical

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

        ''' 
            there are no outliers in this data
            '''