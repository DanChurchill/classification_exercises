import pandas as pd
from sklearn.model_selection import train_test_split




def prep_iris(df):
    '''
    function accepts a dataframe of iris data and performs the cleanup
    operations dictated by the exercises
    '''
    # Drop the species_id and measurement_id columns
    df = df.drop(columns=['species_id','measurement_id'])
    
    # rename the species_name column to species
    df.rename(columns = {'species_name':'species'}, inplace = True)
    
    # Create dummy variables of the species name    
    dummy_df = pd.get_dummies(df['species'], dummy_na=False, drop_first=True)

    # concatenate onto the iris dataframe.
    df = pd.concat([df, dummy_df], axis=1)
    
    # return the converted iris dataframe
    return df



def prep_titanic(df):
    '''
    function accepts a dataframe of titanic data and performs the cleanup
    operations dictated by the exercises
    '''
    # Drop unnecessary columns (pclass)
    df = df.drop(columns=['class', 'deck', 'passenger_id'])

    # Create dummy variables of the categorical columns    
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=False)

    # concatenate onto the titanic dataframe.
    df = pd.concat([df, dummy_df], axis=1)

    # drop rows where age or embarked is missing (Null)
    df = df[df.age.isnull() != True]
    df = df[df.embarked.isnull() != True]

    # return the converted titanic dataframe
    return df

def prep_telco(df):
    '''
    function accepts a dataframe of telco data and performs the cleanup
    operations dictated by the exercises
    '''
def prep_telco(df):
    '''
    function accepts a dataframe of telco data and performs the cleanup
    operations dictated by the exercises
    '''

    # delete index columns
    df = df.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id', 'customer_id'])
    
    # convert total charges to floats
    df.total_charges = df.total_charges.str.replace(' ', '0')
    df.total_charges = df.total_charges.astype(float)
    
    # create dummies for binary columns and concat to df
    dummies = pd.get_dummies(df[['churn', 'gender','senior_citizen', 'partner', 'dependents', 'phone_service',
                                 'paperless_billing']], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
    dummies = pd.get_dummies(df[['multiple_lines', 'online_security', 'online_backup','device_protection',
                                 'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 'payment_type',
                                 'internet_service_type']], drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    
    # delete columns where dummies were created
    df = df.drop(columns=['churn', 'gender','senior_citizen', 'partner', 'dependents', 'phone_service',
                                 'paperless_billing', 'multiple_lines', 'online_security', 'online_backup',
                                 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
                                 'contract_type', 'payment_type','internet_service_type'])
    
    # clean up column names
    df.rename(columns={'churn_Yes' : 'churn', 'gender_Male': 'male', 'partner_Yes' : 'partner',
                       'dependents_Yes':'dependents', 'phone_service_Yes' : 'phone_service', 
                       'paperless_billing_yes':'paperless_billing'}, inplace=True)
    
    return df
    

    
def my_split(df, target):
    '''
    takes a dataframe and a string (the column name of the target)
    returns 3 datframes of train, validate, and test data
    '''
    train_validate, test = train_test_split(df, test_size=.2, stratify=df[target])

    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       stratify=train_validate[target])

    return train, validate, test

