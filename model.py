# importing of neccessary Libraries
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder




# loading data into pandas
df = pd.read_csv('Iris.csv', index_col='Id')

# converting the column label to lower case string
df.columns = df.columns.str.lower()

# Creating inputs and labels for our model
X, y = df.drop(columns='species'), df['species']

# Encoder
target_encoder = LabelEncoder()
target_encoder.fit(y)
y = target_encoder.transform(y)

# Building a Decision-Tree model
model_dt = DecisionTreeClassifier()


