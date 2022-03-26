# importing of neccessary Libraries
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder




# loading data into pandas
df = pd.read_csv('data.csv')

# converting the column label to lower case string
#df.columns = df.columns.str.lower()

df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Creating inputs and labels for our model
X, y = df.drop(columns='diagnosis'), df['diagnosis']

# Encoder
target_encoder = LabelEncoder()
target_encoder.fit(y)
y = target_encoder.transform(y)

# Building a Decision-Tree model
model_dt = LogisticRegression()


