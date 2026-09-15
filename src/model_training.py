import os
import logging
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir = 'logs'
os.makedirs(log_dir , exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join( log_dir,'model_training.log')

file_handler = logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formats)
file_handler.setFormatter(formats)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(param_path:str):
      try:
            with open(param_path , 'r') as file:
                  params = yaml.safe_load(file)
            logger.debug('parameters retrived from %s ',param_path)
            return params
      except FileExistsError:
            logger.error('File not found at %s',param_path)
            raise
      except yaml.YAMLError as e:
            logger.error('YAML error %s ',e)
      except Exception as e:
            logger.error('unexpected error while loading param %s ',param_path)
            raise


def load_data(file_path:str) -> pd.DataFrame:
      """
      load data from a csv file
      """
      try:
            df = pd.read_csv(file_path)
            logger.debug('data loaded from %s with shape %s ',file_path,df.shape ) 
            return df
      except pd.errors.ParserError as e :
            logger.error('failed to parse the csv file %s',e)
            raise
      except FileNotFoundError as e:
            logger.error("file not found %s ",e)
            raise
      except Exception as e:
            logger.error('unexpected error occured while loading the data: %s',e)
            raise
      
def train_model(x_train: np.ndarray , y_train: np.ndarray , params:dict ) -> RandomForestClassifier:
      """
      train a randomforest model on our dataset
      """
      
      try:
            logger.debug('initializing a randomforestclassifier model ')
            clf = RandomForestClassifier(n_estimators=params['n_estimators'] , random_state=params['random_state'])
            clf.fit(x_train,y_train)
            
            logger.debug('model training completed')
            return clf
      except ValueError as e:
            logger.error('value error during model training %s ',e)
            raise
      except Exception as e:
            logger.error('unexpected error occured while model training %s' ,e)
            raise
      
def save_model(model , file_path:str)->None:
      """
      save the trained model to a file
      """
      try:
            os.makedirs(os.path.dirname(file_path),exist_ok=True)
            with open(file_path,'wb') as file:
                  pickle.dump(model , file)
            logger.debug('model saved as pickel file %s ', file_path)
      except FileNotFoundError as e:
            logger.error("file path not found %s",e)
            raise
      except Exception as e:
            logger.error("unexpected error occured while saving model pickle file %s ",e)
            raise

def main():
      try:
            params = load_params(param_path=('params.yaml'))['model_training']
            # params = {'n_estimators':25 , 'random_state':2}
            train_data = load_data('./data/processed/train_tfidf.csv')
            x_train = train_data.iloc[:,:-1].values
            y_train = train_data.iloc[:,-1].values
            
            clf = train_model(x_train,y_train,params)
            
            model_save_path = 'models/model.pkl'
            save_model(clf , model_save_path)
      except Exception as e:
            logger.error('unexpected error occured in model training %s ',e)
            print(f'error : {e}')
            
            
if __name__ == '__main__':
      main()
      
 