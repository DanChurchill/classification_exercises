import pandas as pd
from sklearn.model_selection import train_test_split




def prep_iris(df):
    # Drop the species_id and measurement_id columns
    df = df.drop(columns=['species_id','measurement_id'])
    
    # rename the species_name column to species
    df.rename(columns = {'species_name':'species'}, inplace = True)
    
    # Create dummy variables of the species name    
    dummy_df = pd.get_dummies(df['species'], dummy_na=False, drop_first=False)

    # concatenate onto the iris dataframe.
    df = pd.concat([df, dummy_df], axis=1)
    
    # return the converted iris dataframe
    return df

def prep_titanic(df):
    # Drop unnecessary columns (pclass)
    df = df.drop(columns=['class'])

    # Create dummy variables of the categorical columns    
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=False)

    # concatenate onto the titanic dataframe.
    df = pd.concat([df, dummy_df], axis=1)

    # return the converted titanic dataframe
    return df

def prep_telco(df):
    # Drop unnecessary, unhelpful, or duplicated columns. 
    df = df.drop(columns=['contract_type_id','internet_service_type_id', 'payment_type_id', 'contract_type_id.1',
                          'payment_type_id.1', 'monthly_charges.1','total_charges.1','paperless_billing.1',])
    
    # Create dummy variables of the categorical columns  
    dummy_df = pd.get_dummies(df[['gender','contract_type','internet_service_type']], dummy_na=False, drop_first=False)

    # concatenate onto the telco dataframe.
    df = pd.concat([df, dummy_df], axis=1)

    # return prepared dataframe
    return df
    
def split_titanic(df):
    '''
    Take in a dataframe and return train, validate, and test dataframes
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, stratify=df.survived)

    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       stratify=train_validate.survived)
    return train, validate, test

def split_iris(df):
    '''
    Take in a dataframe and return train, validate, and test dataframes
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, stratify=df.species)

    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       stratify=train_validate.species)
    return train, validate, test

def split_telco(df):
    '''
    Take in a dataframe and return train, validate, and test dataframes
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, stratify=df.churn)

    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       stratify=train_validate.churn)
    return train, validate, test
