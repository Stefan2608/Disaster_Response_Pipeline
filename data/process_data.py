# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        'message_filepath' : path to a csv file
        'categories_filepath' : path to a csv file
    OUTPUT:
        transformed pandas dataframe
    '''
    # load datasets
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    

    
    return df
     

def clean_data(df):
    '''
    INPUT:
        dataframe
        
    OUTPUT:
        cleaned dataframe 
       
    '''     
    # categorie split 
    categories = df['categories'].str.split(pat=';', expand = True)
                    
    # select first row
    row = categories.iloc[[1]]
    # extract columnsnames

    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
                    
    # rename columnsnames
    categories.columns = category_colnames
                    
    # converting categories
    for column in categories:  
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(np.int)
    
    
    # drop original categories
    df = df.drop('categories', axis=1)

    # concate the dataframe
    df = pd.concat([df, categories], axis=1)
  
    # drop dublicates
    df = df.drop_duplicates()     
    
    # dropping the 2 value from column related
    df.loc[df['related'] > 1,'related'] = 0
        
    return df


def save_data(df, database_filename):
    
    '''
    INPUT:
          df = dataframe
        - cleaned dataframe ready to move to sql
        - string of desired database name, in format 'databasename.db'
       
    OUTPUT:
        - None
    '''

    # create sqlite engine
    engine = create_engine('sqlite:///'+ database_filename)

    # send df to sqlite file, omitting the index
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    
    # print statement
    print("Data was saved to {}".format(database_filename))

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database.')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
