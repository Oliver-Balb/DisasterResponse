import sys
import nltk
# nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine

database_filepath = 'sqlite:///drp.db'


def load_data(database_filepath):
    '''
    Load data (messages and labels) from sqlite db into dataframe df
    INPUT:
    database_filepath - path and filename of database instance

    OUTPUT:
    X - cleaned messages 
    Y - catagories (labels)
    cat_names - Names of categories (column names)
    '''
    # load data from database

    # Create an engine to connect to the SQLite database
    engine = create_engine(database_filepath)

    # Read the table into a DataFrame
    df = pd.read_sql_table('msgcat', con=engine)
    
    # Split data into message texts and labels (categories) 
    X = df['message']
    Y = df.drop(labels=['id', 'message', 'original', 'genre'], axis=1)
    cat_names = Y.columns
    #print(f"Y.columns.size: {Y.columns.size}")

    return X, Y, cat_names


def tokenize(text):
    '''
    Standardize single message (text) for ML by tokenization
    INPUT:
    text - single text cell

    OUTPUT:
    clean_tokens: tokens generated from text
    '''   
  
    url_regex1 = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_regex2 = r'http\s(?:bit.ly|ow.ly)\s\S+'

    # Standardize meaningless URLs
    detected_urls = []
    detected_urls1 = re.findall(url_regex1, text)
    detected_urls2 = re.findall(url_regex2, text)

    detected_urls = detected_urls1 + detected_urls2
        
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Create tokens
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Define pipeline for data processing, modeling steps
    and training. 
    INPUT:
    -
    OUTPUT:
    model - prediction model
    '''   
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameter grid for GridSearchCV
    param_grid = {
        'clf__estimator__n_estimators': [50, 100],  
        'clf__estimator__max_depth': [None, 10, 20],
        'clf__estimator__min_samples_split': [2, 5],
    }

    # Instantiate GridSearchCV
    model = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model on test data
    INPUT:
    model - 
    X_test - messages
    Y_test - categories (labels)
    category_names - Names of labels
    
    OUTPUT:
    -
    '''  

    # Predict on test data
    Y_pred = model.predict(X_test)
    
    cr = classification_report(Y_test.values, Y_pred, target_names=category_names)
    print(cr)
    

def save_model(model, model_filepath):
    '''
    Save model using pickle to file system
    INPUT:
    model - prediciton model
    model_filepath - path and filename of pickle file (pkl-extension!)
    OUTPUT:
    -
    '''   

    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)

def main1():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


def main2():
    database_filepath= 'sqlite:///./data/drp.db' 
    model_filepath = './models/drp_classifier.pkl'
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

if __name__ == '__main__':
    # main1()
    main2()
