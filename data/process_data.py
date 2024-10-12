import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load training and test data (messages and labels) in CSV format from filesystem
    INPUT:
    messages_filepath - path and filename of messages csv file
    categories_filepath - path and filename of categories csv file
    
    OUTPUT:
    df - messages inner join categories on messages.id = categories.id
    '''
       
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(categories, on='id', how='inner')   
    
    return df

def clean_data(df):
    '''
    Transform training and test data (messages and labels)
    INPUT:
    df - messages and categories dataframe
    OUTPUT:
    df - cleaned messages and categories dataframe
    '''
    # Split `categories` into separate category columns:
    # create a dataframe of the 36 individual category columns
    categories = pd.concat([df['id'], df['categories'].str.split(pat=';', expand=True)], axis=1)


    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = ['id'] + row[1:].apply(lambda x: x[:len(x)-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    # Convert category values to just numbers 0 or 1:
    # Exclude the 1st column (id) from conversion
    columns = (categories.columns.tolist())[1:]

    for column in columns:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x)[-1] if pd.notnull(x) else x)
        
    # convert columns from string to numeric type
    categories = categories.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
    
    # drop the original categories column from `df`
    df.drop(labels=['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, on='id', how='inner')
          
    # drop duplicates
    df = df.drop_duplicates(subset=df.columns.difference(['id']))
   
    # drop 193 rows with 'related' == 2 values
    # Identify the rows to drop
    rows_to_drop = df[df['related'] == 2].index

    # Drop the rows
    df.drop(rows_to_drop, inplace=True)
      
    # drop empty column 'child_alone'
    df.drop(columns=['child_alone'], inplace = True)
    return df
    

def save_data(df, database_filepath):
    '''
    Save cleaned data in dataframe df to specified sqlite db
    INPUT:
    df - cleaned messages and labels data
    database_filepath - path and filename of sqlite db
    
    OUTPUT:
    '''
    engine = create_engine(database_filepath)
    df.to_sql('msgcat', engine, index=False, if_exists='replace')


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
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
            'datasets as the first and second argument respectively, as '\
            'well as the filepath of the database to save the cleaned data '\
            'to as the third argument. \n\nExample: python process_data.py '\
            'disaster_messages.csv disaster_categories.csv '\
            'DisasterResponse.db')
       
if __name__ == '__main__':
     main()