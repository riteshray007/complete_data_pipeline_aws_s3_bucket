import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import logging
import yaml

log_dir = 'logs'
os.makedirs(log_dir , exist_ok=True )

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_logger = logging.StreamHandler()
console_logger.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_logger = logging.FileHandler(log_file_path)
file_logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_logger.setFormatter(formatter)
file_logger.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(file_logger)

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


def load_data(file_path:str)-> pd.DataFrame:
      try:
            df = pd.read_csv(file_path)
            df.fillna("" , inplace=True )
            logger.debug("data loaded and naans filled from %s" , file_path)
            return df
      except pd.errors.ParserError as e:
            logger.error("failed to parse the csv file %s ", e)
            raise
      except Exception as e:
            logger.error("Unexpected error occured while loading the data %s ", e)
            raise
      
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        x_train = train_data["text"].values
        y_train = train_data["target"].values
        x_test = test_data["text"].values
        y_test = test_data["target"].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        tfidf = TfidfTransformer()
        x_train_tfidf = tfidf.fit_transform(x_train_bow)
        x_test_tfidf = tfidf.transform(x_test_bow)

        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df["label"] = y_test

        logger.debug("TF-IDF applied and data transformed")
        return train_df, test_df

    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise
  
      
def save_data(df:pd.DataFrame , file_path:str)-> None:
      """
      save the dataframe to a csv file .
      """
      try:
            os.makedirs(os.path.dirname(file_path),exist_ok=True)
            df.to_csv(file_path,index=False)
            logging.debug('data saved inside %s',file_path)
      except Exception as e:
            logging.error("unexpected error occured while saving the data %s ",e)
            raise

def main(): 
      try:
            params = load_params(param_path=('params.yaml'))
            max_features = params['feature_engineering']['max_features']
            
            train_data = load_data('./data/interim/train_processed.csv')
            test_data = load_data('./data/interim/test_processed.csv')
            
            train_df , test_df = apply_tfidf(train_data,test_data,max_features)
            save_data(train_df , os.path.join('./data','processed','train_tfidf.csv'))
            save_data(test_df , os.path.join('./data','processed','test_tfidf.csv'))
            
            logging.debug('data saved at %s ' , os.path.join('./data','processed') )
      except Exception as e:
            logging.error('unexpected error occured while feature engineering the data %s',e)
            print(f"Error: {e}") 
      
if __name__ == '__main__':
      main()
      
      
      