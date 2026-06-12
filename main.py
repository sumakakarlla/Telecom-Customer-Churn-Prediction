'''
In this file we are going to load the customer_retention-data and pass the data to all ML pipeline
to built complete Machine learning system
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings("ignore")
import logging
import sys
from log_code import setup_logging
logger =setup_logging('main')
from sklearn.model_selection import train_test_split
from missing_data_handling import missing
from VT import var_transformation
from FS import feature_sel
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_models import common

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
class Customer_Retention_Prediction:
    def __init__(self,path):
        try:
            self.data = pd.read_csv(path)
            self.path = path
            self.data = pd.read_csv(self.path)
            self.data = self.data.replace(r'^\s*$',np.nan,regex=True)
            logger.info(f"The shape of the data before adding sim column was : {self.data.shape}")
            logger.info(f"the no. of null values  are :{self.data.isnull().sum()}")
            #adding sim column
            self.sim_values=['jio','airtel','vodaphone','bsnl']
            self.data['sims']= [self.sim_values[i%4] for i in range(len(self.data))]
            logger.info(f"The shape of the data after adding sim column was : {self.data.shape}")

            logger.info(self.data.dtypes)
            self.data['TotalCharges']=pd.to_numeric(self.data['TotalCharges'])#converting totalcharges column from str to numeric
            self.data['Churn'] =self.data['Churn'].map({'Yes':1,'No':0}).astype(int)

            logger.info(f'the count of Churn labels :{self.data['Churn'].value_counts()}')
            logger.info(f'the datatypes of columns are :{self.data.dtypes}')


            self.X=self.data.drop(['Churn'],axis=1) #independent columns
            self.y=self.data['Churn'] #dependent columns
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f"The shape of the training data is : {self.X_train.shape}=> {self.y_train.shape}")
            logger.info(f"The shape of the testing data is : {self.X_test.shape}=> {self.y_train.shape}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")
    def handling_missing_values(self):
        try:
            logger.info(f"the no. of null values in X_train are :{self.X_train.isnull().sum()}")
            logger.info(f"the no. of null values in X_test  are :{self.X_test.isnull().sum()}")
            '''
            we are implementing  mode technique
             '''
            self.X_train,self.X_test= missing(self.X_train , self.X_test)
            logger.info(f"the no. of null values after mode technique in X_train are :{self.X_train.isnull().sum()}")
            logger.info(f"the no. of null values after mode technique in X_test  are :{self.X_test.isnull().sum()}")

            '''
                        now we are  separating X_train and X_test into numerical and categorical
            '''


            # logger.info(f"{self.X_train.info()}")
            self.X_train_numerical = self.X_train.select_dtypes(exclude='str')
            self.X_train_categorical = self.X_train.select_dtypes(include='str')
            self.X_test_numerical = self.X_test.select_dtypes(exclude='str')
            self.X_test_categorical = self.X_test.select_dtypes(include='str')
            logger.info("============X_train_seperation_details===================")
            logger.info(f"X_train : {self.X_train.shape}")
            logger.info(f"X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"X_train_categorical : {self.X_train_categorical.shape} : \n : {self.X_train_categorical.columns}")
            logger.info(f"==========X_test_seperation_details=======================")
            logger.info(f"X_test : {self.X_test.shape}")
            logger.info(f"X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(f"X_test_categorical : {self.X_test_categorical.shape} : \n : {self.X_test_categorical.columns}")
            '''
            From the above data we got to know that in training total we have 21 columns in 17 we have 4 num and 16 cat
            this info is both for training data (X_train) and also for testing data (X_test)
            '''

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def Variable_Transformation(self):
        try:
            logger.info(f' train_numerical data shape and columns : {self.X_train_numerical.shape} :\n : {self.X_train_numerical.columns}')
            logger.info(f'the shape and columns of test_numerical data is : {self.X_test_numerical.shape} :\n : {self.X_test_numerical.columns}')
            self.X_train_numerical,self.X_test_numerical=var_transformation(self.X_train_numerical,self.X_test_numerical)
            logger.info(f"After X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"After X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def feature_selection(self):
        try:
            logger.info(f"Before Feature Selection : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"Before Feature Selection : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            self.X_train_numerical , self.X_test_numerical =feature_sel(self.X_train_numerical, self.X_test_numerical,self.y_train)
            logger.info(f"After Feature Selection : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"After Feature Selection : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def Cat_Num(self):
        try:
            logger.info(f" Categorical Columns : {self.X_train_categorical.columns}")
            '''
                    dropping CustomerId as it is not needed
            '''
            self.X_train_categorical = self.X_train_categorical.drop('customerID', axis=1)
            self.X_test_categorical = self.X_test_categorical.drop('customerID', axis=1)

            '''
            applying one_hot encoding to gender , paperlessBilling , PaymentMethod and sims column
            '''
            one_hot_obj = OneHotEncoder(drop="first")
            one_hot_obj.fit(self.X_train_categorical[["gender", "PaperlessBilling","PaymentMethod","sims"]])
            values_1 = one_hot_obj.transform(self.X_train_categorical[["gender", "PaperlessBilling","PaymentMethod","sims"]]).toarray()
            values_2 = one_hot_obj.transform(self.X_test_categorical[["gender", "PaperlessBilling","PaymentMethod","sims"]]).toarray()
            t1 = pd.DataFrame(data=values_1, columns=one_hot_obj.get_feature_names_out())
            t2 = pd.DataFrame(data=values_2, columns=one_hot_obj.get_feature_names_out())
            self.X_train_categorical.reset_index(drop=True, inplace=True)
            self.X_test_categorical.reset_index(drop=True, inplace=True)
            t1.reset_index(drop=True, inplace=True)
            t2.reset_index(drop=True, inplace=True)
            self.X_train_categorical_new = pd.concat([self.X_train_categorical, t1], axis=1)
            self.X_test_categorical_new = pd.concat([self.X_test_categorical, t2], axis=1)
            self.X_train_categorical_new = self.X_train_categorical_new.drop(["gender", "PaperlessBilling","PaymentMethod","sims"], axis=1)
            self.X_test_categorical_new = self.X_test_categorical_new.drop(["gender", "PaperlessBilling","PaymentMethod","sims"], axis=1)
            '''
            applying odinal encoding to 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract' columns
            '''
            odi_obj = OrdinalEncoder()
            odi_obj.fit(self.X_train_categorical[['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']])
            values_3 = odi_obj.transform(self.X_train_categorical[['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']])
            values_4 = odi_obj.transform(self.X_test_categorical[['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']])

            t3 = pd.DataFrame(data=values_3, columns=odi_obj.get_feature_names_out() + "_odinal")
            t4 = pd.DataFrame(data=values_4, columns=odi_obj.get_feature_names_out() + "_odinal")

            t3.reset_index(drop=True, inplace=True)
            t4.reset_index(drop=True, inplace=True)

            self.X_train_categorical_new.reset_index(drop=True, inplace=True)
            self.X_test_categorical_new.reset_index(drop=True, inplace=True)

            self.X_train_categorical_new = pd.concat([self.X_train_categorical_new, t3], axis=1)
            self.X_test_categorical_new = pd.concat([self.X_test_categorical_new, t4], axis=1)

            self.X_train_categorical_new = self.X_train_categorical_new.drop(
                ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract'], axis=1)
            self.X_test_categorical_new = self.X_test_categorical_new.drop(['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract'], axis=1)

            logger.info(f"After Converting categorical data to numerical")
            logger.info(
                f"X_train_cat_data : {self.X_train_categorical_new.shape}: \n : {self.X_train_categorical_new.columns} : \n : {self.X_train_categorical_new.isnull().sum()}")
            logger.info(
                f"X_test_cat_data : {self.X_test_categorical_new.shape}: \n : {self.X_test_categorical_new.columns} : \n : {self.X_test_categorical_new.isnull().sum()}")

            '''
            we converted all categorical columns to numeric, now we have to  merge these converted categorical columns and numeric columns
            '''
            self.X_train_numerical.reset_index(drop=True, inplace=True)
            self.X_test_numerical.reset_index(drop=True, inplace=True)

            self.X_train_categorical_new.reset_index(drop=True, inplace=True)
            self.X_test_categorical_new.reset_index(drop=True, inplace=True)

            self.final_training_data = pd.concat([self.X_train_numerical, self.X_train_categorical_new], axis=1)
            self.final_testing_data = pd.concat([self.X_test_numerical, self.X_test_categorical_new], axis=1)

            logger.info(f"========Final dataset================")
            logger.info(f"Final Train Independent data")
            logger.info(
                f"{self.final_training_data.shape} : \n : {self.final_training_data.columns} : \n : {self.final_training_data.isnull().sum()}")
            logger.info(f"Final Test Independent data")
            logger.info(
                f"{self.final_testing_data.shape} : \n : {self.final_testing_data.columns} : \n : {self.final_testing_data.isnull().sum()}")




        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")
    '''
    checking if train and test data is balanced or not . if data is not balanced we will use 'Over Sampling' technique to balance data'''
    def balancing(self):
        try:
            logger.info(f"===== Checking Data is Balanced or Not========")
            logger.info(f"Number of rows for Good Customers : {1} -> : {sum(self.y_train == 1)}")
            logger.info(f"Number of rows for Bad Customers : {0} -> : {sum(self.y_train == 0)}")
            smote_reg = SMOTE(random_state=42)
            self.final_training_data_bal, self.y_train_bal = smote_reg.fit_resample(self.final_training_data,
                                                                                  self.y_train)
            logger.info(f"=====After Checking Data is Balanced or Not========")
            logger.info(f"Number of rows for Good Customers : {1} -> : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of rows for Bad Customers : {0} -> : {sum(self.y_train_bal == 0)}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def scaling_data(self):
        try:
            print(self.final_training_data_bal)
            sc = StandardScaler()
            sc.fit(self.final_training_data_bal)
            self.final_training_data_scaled = sc.transform(self.final_training_data_bal)
            self.final_testing_data_scaled = sc.transform(self.final_testing_data)
            print(self.final_training_data_scaled)
            common(self.final_training_data_scaled,self.y_train_bal,self.final_testing_data_scaled,self.y_test)
            logger.info("===Training LogisticRegression =====")
            self.model = LogisticRegression()
            self.model.fit(self.final_training_data_scaled, self.y_train_bal)
            self.y_test_predictions = self.model.predict(self.final_testing_data_scaled)
            logger.info(f"Test data Accuracy : {accuracy_score(self.y_test, self.y_test_predictions)}")
            logger.info(f"Confusion Matrix : {confusion_matrix(self.y_test, self.y_test_predictions)}")
            logger.info(f"classification report : {classification_report(self.y_test, self.y_test_predictions)}")
            logger.info(f"====Saving the Scaled and Logistic Regression Model into Pickle File===========")
            with open("standard_scaler.pkl", "wb") as f:
                pickle.dump(sc, f)

            with open("Credit_card_APP.pkl", "wb") as f1:
                pickle.dump(self.model, f1)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")



if __name__ == "__main__":
    try:
        obj = Customer_Retention_Prediction("Telco-Customer-Churn.csv")
        obj.handling_missing_values()
        obj.Variable_Transformation()
        obj.feature_selection()
        obj.Cat_Num()
        obj.balancing()
        obj.scaling_data()
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")