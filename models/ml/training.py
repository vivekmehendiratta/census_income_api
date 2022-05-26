import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier as xgb_clf
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score
from sklearn.model_selection import cross_validate

from models.ml.data_definition import train_cols, test_cols, education_dict, other_cols

def transform_data(data, request_type='model_building'):
    '''
    input:
        None
    output:
        X: Training message Lis.
        y: Training target
    '''
    # df = pd.read_csv(StringIO(response))

    # add column names for better understanding
    if request_type == 'predict':
        data.columns = test_cols
        data.drop('instance_weight', axis = 1, inplace=True)

        # strip all object variables
        data=data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        data.replace('?', np.nan,inplace=True)

        for col in other_cols:
            data[col] = data[col].fillna('Other')

        data['education'].map(lambda x: education_dict[x.strip()])
        data['occupation_code'] = data['occupation_code'].astype(object)
        data['industry_code'] = data['industry_code'].astype(object)
        
        X = data
        y = []
    else:
        data.columns = train_cols
        data.drop('instance_weight', axis = 1, inplace=True)

        # strip all object variables
        data=data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        data.replace('?', np.nan,inplace=True)

        for col in other_cols:
            data[col] = data[col].fillna('Other')

        data['education'].map(lambda x: education_dict[x.strip()])
        data['occupation_code'] = data['occupation_code'].astype(object)
        data['industry_code'] = data['industry_code'].astype(object)

        data['income'] = data['income'].map(lambda x: 0 if x=='- 50000.' else 1)

        y = data['income']
        X = data.drop('income', axis = 1)

    return X, y

def build_model(X, y):
    '''
    input:
        X, y
    output:
        pipeline
    '''

    # Preprocess pipeline
    ## missing value imputation
    ## Scaling
    ## One hot encoding

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehotencoder', OneHotEncoder(handle_unknown='ignore',sparse=False))
    ])

    numerical_features = X.select_dtypes(include=['int64','float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers = [
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # XGBoost Classifier

    pipe = Pipeline([
        ('preprocessor',preprocessor),
        ('pca', PCA(n_components=20)),
        ('classifier', xgb_clf())
    ])

    pipe.fit(X, y)

    return pipe

def evaluate_model(model, X, y, cv=5):

    scores = cross_validate(model, X, y, cv=cv, scoring={
            "precision" : make_scorer(precision_score, average = 'weighted'),
            "recall" : make_scorer(recall_score, average = 'weighted'),
            "f1_score" : make_scorer(f1_score, average='weighted')
            })
    
    return scores

def save_model(model, file_name):
    joblib.dump(model, file_name)