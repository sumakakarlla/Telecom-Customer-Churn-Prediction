import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from log_code import setup_logging
logger = setup_logging("missing_data_handling")

def missing(X_train , X_test):
    try:
        logger.info(f"the no. of null values in X_train are :{X_train.isnull().sum()}")
        logger.info(f"the no. of null values in X_test are :{X_test.isnull().sum()}")
        X_train['TotalCharges'] = X_train['TotalCharges'].fillna(X_train['TotalCharges'].mode()[0])
        X_test['TotalCharges'] = X_test['TotalCharges'].fillna(X_test['TotalCharges'].mode()[0])
        logger.info(f"the no. of null values after mode technique in X_train are :{X_train.isnull().sum()}")
        logger.info(f"the no. of null values after mode technique in X_test are :{X_test.isnull().sum()}")
        return X_train, X_test



    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

