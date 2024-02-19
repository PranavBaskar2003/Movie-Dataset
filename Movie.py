import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from collections import namedtuple
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
data = pd.read_csv('c:/Users/Pranav/Documents/Movie.csv', encoding="Latin1").dropna()
data['Duration'] = data['Duration'].str.replace('min', '').astype(float)
data["Votes"] = data["Votes"].replace("$5.16M", 516)
data["Votes"] = data['Votes'].str.replace(',', '').astype(float)
data.dropna(subset=['Year'], inplace=True)
data.dropna(subset=['Genre'], inplace=True)
data['Rating'] = data['Rating'].fillna(data['Rating'].mode().max())
data['Duration'] = data['Duration'].fillna(data['Duration'].mean())
data['Votes'] = data['Votes'].fillna(data['Votes'].mean())
print(data.head())
print(data.info())
print(data.shape)
print(data.describe())
print(data['Genre'].nunique())
print(data['Genre'].head())
print(data.Year.unique())
print(data.Rating.unique())  
print(data['Duration'].unique())
print(data.groupby(['Genre']).count())
print(data["Director"].value_counts().head(6))
print('Null Values in Year Column:', data['Year'].isnull().sum())
print('Null Values in Genre Column:', data['Genre'].isnull().sum())
print('Number of duplicate rows:', data.duplicated().sum())
years = data['Year']
ratings = data['Rating']

def awesome(column):
    global data
    data[column].value_counts().sort_values(ascending=False)[:10].plot(kind="bar", figsize=(20,6), edgecolor="k", color="green")
    plt.xticks(rotation=0)
    plt.title("Top Ten {}".format(column))
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()
awesome("Director")
awesome("Actor 1")
awesome("Actor 2")
awesome("Actor 3")

data_year_rating = pd.DataFrame({'Year': years, 'Rating': ratings})
count_movies = data_year_rating.groupby(['Year', 'Rating']).size().reset_index(name='Count')
plt.figure(figsize=(24, 16))
sns.barplot(x='Year', y='Count', hue='Rating', data=count_movies, palette='plasma')
plt.title('Number of Movies within a Year Based on Rating')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.legend(title='Rating', loc='upper right')
plt.show()

genres_columns = ['Genre']
genre_counts = data['Genre'].value_counts(normalize=True) * 100
main_genres = genre_counts[genre_counts > 0.75].index
data['Genre'] = data['Genre'].apply(lambda x: x if x in main_genres else 'Others')
label = data['Genre'].value_counts().index
sizes = data['Genre'].value_counts()
plt.figure(figsize=(10, 10))
plt.pie(sizes, labels=label, startangle=0, shadow=False, autopct='%1.1f%%')
plt.title('Movie Genres')
plt.show()
