#import libriaries 
import sys
import re
import pickle
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine

nltk.download('stopwords')

def load_data(database_filepath):
    """ 
    Input: 
    data_filepath - String
                  
    Output: 
    x - pandas df: messages 
    y - pandas df: validated categories    
    lables - list: classification list of message
    
    """
    # load data 
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]
    label = list(df.columns[4:])
    
    return X, y, label
 
              
def tokenize(text):
    """ 
    Input: 
    text: text to tokenize 
    
    Output:         
    clean words: list of words after tokennize and lemmatize
    """
    
    # text cleaning          
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())    
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]        
    
    #lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_words = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    clean_words = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean_words]
              
    return clean_words          
    
def build_model():
    """ 
    Input: 
    Parameters for classification
    
    Output:         
    grid: Gridsearch 
    
    """    
    # Model definition
    classification = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', classification)
        ])

    parameters = {'clf__estimator__max_depth': [10, 30, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}
    
    grid = GridSearchCV(pipeline, parameters) 
               
    return grid    
               


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model performance
    Input:
    
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for categories
    Output:
        print: Performance of the model
    '''
    
   #Model evaluation
    Y_pred = model.predict(X_test)   


    #Print out the accuracy
    for i in range(len(category_names)):
        print("categories:", category_names[i],"\n", classification_report(Y_test.iloc[:,i].values, Y_pred[:,i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:,i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    """
    Save  function
    
    This function saves trained mode file
    
    Input:
        model: model to be saved 
        model_filepath: path of the file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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


if __name__ == '__main__':
    main()
