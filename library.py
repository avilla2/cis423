import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin #gives us the tools to build custom transformers

#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
    def __init__(self, mapping_column, mapping_dict:dict):  
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column  #column to focus on

    def fit(self, X, y = None):
        print("Warning: MappingTransformer.fit does nothing.")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
        X_ = X.copy()
        X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result


class OHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, dummy_na=False, drop_first=True):  
        self.target_column = target_column
        self.dummy_na = dummy_na
        self.drop_first = drop_first
    
    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        return pd.get_dummies(X,
                            prefix=self.target_column,    #your choice
                            prefix_sep='_',     #your choice
                            columns=[self.target_column],
                            dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                            drop_first=self.drop_first    #really should be True but I wanted to give a clearer picture
                            )

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result

    def fit(self, X, y = None):
        print("Warning: MappingTransformer.fit does nothing.")
        return X


class RenamingTransformer(BaseEstimator, TransformerMixin):
    #First write __init__ method.
    #Hint: maybe copy and paste from MappingTransformer here then fix up for new problem?
    def __init__(self, mapping_dict:dict):  
        self.mapping_dict = mapping_dict
    
    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
        verify = [ k for k in self.mapping_dict.keys() if k not in X.columns.to_list()]
        assert len(verify) == 0, f'MappingTransformer.transform unknown column(s) {verify}'
        X_ = X.copy()
        X_.rename(columns=self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result

    def fit(self, X, y = None):
        print("Warning: MappingTransformer.fit does nothing.")
        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_list, action='drop'):
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list = column_list
        self.action = action

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        if self.action == 'drop':
            X_ = X.drop(columns=self.column_list)
        else:
            X_ = X[self.column_list]
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result

    def fit(self, X, y = None):
        print("Warning: MappingTransformer.fit does nothing.")
        return X


class Sigma3Transformer(BaseEstimator, TransformerMixin):
  
    def __init__(self, target_column):  
        self.target_column = target_column

    def fit(self, X, y = None):
        print("Warning: MappingTransformer.fit does nothing.")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])
        X_ = X.copy()
        #compute mean of column - look for method
        m = X_[self.target_column].mean()
        #compute std of column - look for method
        sigma = X_[self.target_column].std()
        minb, maxb = (m - 3 * sigma, m + 3 * sigma)
        X_[self.target_column] = X_[self.target_column].clip(lower=minb, upper=maxb)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result


class TukeyTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, target_column, fence='outer'):
        assert fence in ['inner', 'outer']
        self.target_column = target_column
        self.fence = fence

    def fit(self, X, y = None):
        print("Warning: MappingTransformer.fit does nothing.")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])
        X_ = X.copy()
        q1 = X_[column].quantile(0.25)
        q3 = X_[column].quantile(0.75)
        iqr = q3-q1
        outer_low = q1-3*iqr
        outer_high = q3+3*iqr
        inner_low = q1-1.5*iqr
        inner_high = q3+1.5*iqr
        if self.fence == 'inner':
            X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
        else:
            X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result  
    