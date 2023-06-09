import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from embedding_encoder import EmbeddingEncoder
from embedding_encoder.utils.compose import ColumnTransformerWithNames
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

random.seed(42)

df = pd.read_csv('dummy.csv')

Y = df["Label"]
X = df.drop(["Label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# feature categories
categorical = ["Color"]
numeric = ["Num1", "Num2", "Num3", "Num4"]

# ee = EmbeddingEncoder(task="classification", mapping_path="mapping.txt")
# num_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
# cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), ee)
# col_transformer = ColumnTransformer([("num_transformer", num_pipe, numeric), ("cat_transformer", cat_pipe, categorical)])

# pipe = make_pipeline(col_transformer, LogisticRegression())      
# pipe.fit(X_train, y_train)       
# pred = pipe.predict(X_test)     

ee = EmbeddingEncoder(task="classification", mapping_path="mapping.txt")
scaler = StandardScaler()
imputer = ColumnTransformerWithNames([("numeric", SimpleImputer(strategy="mean"), numeric),
                                        ("categorical", SimpleImputer(strategy="most_frequent"),
                                        categorical)])
processor = ColumnTransformer([("embeddings", ee, categorical),
                                ("scale", scaler, numeric)])

pipe = make_pipeline(imputer, processor, LogisticRegression())
pipe.fit(X_train, y_train)       
pred = pipe.predict(X_test)   

ee = pipe.named_steps["columntransformer"].named_transformers_["embeddings"]

plot = ee.plot_embeddings(variable="Color", model="pca")
plt.show()