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

    # Drop unnecessary, unhelpful, or duplicated columns. 
    df = df.drop(columns=['contract_type_id','internet_service_type_id', 'payment_type_id', 'contract_type_id.1',
                          'payment_type_id.1', 'monthly_charges.1','total_charges.1','paperless_billing.1'])

    # Create dummy variables of the categorical columns  
    dummy_df = pd.get_dummies(df[['gender','contract_type','internet_service_type']], dummy_na=False, drop_first=False)

    # concatenate onto the telco dataframe.
    df = pd.concat([df, dummy_df], axis=1)

    # return prepared dataframe
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

