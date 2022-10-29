import pandas as pd
import uuid
import pickle
import numpy as np
import warnings
# Scaling
from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")


class BaseModel():
    def __init__(self, model_name, model_params={}):

        self.model_name = model_name
        self.model_id = uuid.uuid4()
        self.model_is_trained = False

        if self.model_name not in ['svc', 'logreg']:
            raise TypeError('Select other model: svc or logreg')

        if self.model_name == 'svc':
            self.model = SVC(**model_params)
        elif self.model_name == 'logreg':
            self.model = LogisticRegression(**model_params)


        # try:
        #     if self.model_name == 'svc':
        #         self.model = SVC(**model_params)
        #     elif self.model_name == 'logreg':
        #         self.model = LogisticRegression(**model_params)
        # except Exception as e:
        #     raise TypeError(f"Bad params, choose another")

        try:
            self.df = pd.read_csv("/Users/eandrianova/PycharmProjects/pythonProject5/data/heart.csv")
        except Exception:
            raise TypeError("No data, check path")

        # preprocessing
        cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
        con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

        self.X = self.df.drop(['output'], axis=1)
        self.y = self.df[['output']]

        self.scaler = RobustScaler()
        self.X[con_cols] = self.scaler.fit_transform(self.X[con_cols])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        msg = f'Model {self.model_name} {self.model_id} fitted'
        self.model_is_trained = True
        return msg

    def predict(self):
        if self.model_is_trained:
            return self.model.predict(self.X_test)
        else:
            raise TypeError(f"Not fitted model, train on previous step")
