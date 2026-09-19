import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score , precision_score , recall_score, roc_auc_score
import yaml
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir , 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formats)
console_handler.setFormatter(formats)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(param_path:str):
      try:
            with open(param_path , 'r') as file:
                  params = yaml.safe_load(file)
            logger.debug('parameters retrived from %s ',param_path)
            logger.debug('type of param %s',type(params['data_ingestion']['test_size']))
            return params
      except FileNotFoundError:
            logger.error('File not found at %s',param_path)
            raise
      except yaml.YAMLError as e:
            logger.error('YAML error %s ',e)
            raise
      except Exception as e:
            logger.error('unexpected error while loading param %s ',param_path)
            raise 

def load_model(file_path:str):
      """
      load trained model from a file
      """
      try:
            with open(file_path,'rb') as file:
                  model = pickle.load(file)
            logger.debug('model loaded from %s', file_path)
            return model
      except FileNotFoundError as e:
            logger.error("file not found %s ",e)
            raise
      except Exception as e:
            logger.error("unexpected error occurred while loading the model %s",e)
            raise
      
def load_data(file_path:str)->pd.DataFrame:
      """load data from a csv file"""
      try:
            df = pd.read_csv(file_path)
            return df
      except FileNotFoundError as e:
            logging.error('file not found at: %s',e)
            raise
      except Exception as e:
            logging.error('unexpected error while loading data %s',e)
            raise
      
def evaluate_model(clf , x_test , y_test ):
      """evaluate the model and return the evaluation matrics"""
      try:
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)[:,1]
            
            accuracy = accuracy_score(y_test , y_pred)
            precision = precision_score(y_test , y_pred)
            roc_auc = roc_auc_score(y_test , y_pred)
            recall = recall_score(y_test , y_pred)
            
            metrics_dict = {
                  'accuracy':accuracy,
                  'precision':precision,
                  'roc_auc':roc_auc,
                  'recall':recall
            }
            logger.debug('Model evaluation metrics calculated')
            return metrics_dict
      except Exception as e:
            logger.error('unable to evaluate models matrics %s',e)
            raise
      
def save_metrics(metrics:dict , file_path:str )-> None:
      """save the evaluation metrics to a json file"""
      try:
            os.makedirs(os.path.dirname(file_path),exist_ok=True)
            
            with open(file_path,'w') as file:
                  json.dump(metrics,file,indent=4)
            logger.debug('metrics saved to %s ',file_path)
      except Exception as e:
            logger.error('error occurred while saving json %s ',e)
            raise

def main():
      try:
            params = load_params(param_path='params.yaml')
            clf  = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_tfidf.csv')
            train_data = load_data('./data/processed/train_tfidf.csv')
            
            x_train = train_data.iloc[:,:-1].values
            y_train = train_data.iloc[:,-1].values
            x_test = test_data.iloc[:,:-1].values
            y_test = test_data.iloc[:,-1].values
            
            train_metrics = evaluate_model(clf , x_train , y_train)
            metrics = evaluate_model(clf,x_test , y_test)
            
            with Live(save_dvc_exp=True) as dvclive:
                  dvclive.log_metric("train_accuracy", train_metrics["accuracy"])
                  dvclive.log_metric("train_precision", train_metrics["precision"])
                  dvclive.log_metric("train_recall", train_metrics["recall"])
                  dvclive.log_metric("accuracy", metrics["accuracy"])
                  dvclive.log_metric("precision", metrics["precision"])
                  dvclive.log_metric("recall", metrics["recall"])
                  
                  dvclive.log_params(params)              
            
            save_metrics(metrics,'reports/metrics.json')
            
      except Exception as e:
            logger.error('error occurred while model_evaluation %s ',e)
            print(f'error: {e}')
            

if __name__  == '__main__':
      main()

      


