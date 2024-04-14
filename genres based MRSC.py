#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


credits.head()


# In[4]:


credits.shape


# In[5]:


credits.info()
credits.isnull().sum() #perform for no null value elimination of error of data entry


# In[6]:


movies.isnull().sum()


# In[7]:


credits.columns = ['id','title','cast','crew']
movies = movies.merge(credits , on = 'id')


# In[8]:


movies.head(5)


# In[9]:


c = movies['vote_average'].mean() #Voted for mean 
c


# In[10]:


m = movies['vote_count'].quantile(0.9)  #Minimum Vote applied
m


# In[11]:


minvote = 100
movies_list = movies.copy().loc[movies['vote_count']>= m]
movies_list.shape


# In[12]:


def weighted_rating(x,m=m,c=c):
    v = x['vote_count']
    r = x['vote_average']
    return (v/(v+m)*r)+(m/(m+v)*c)


# 

# In[13]:


movies_list['score'] = movies_list.apply(weighted_rating, axis = 1)


# In[14]:


movies_list.head()


# In[15]:


# Sort the DataFrame by 'score' column in descending order
movies_list = movies_list.sort_values('score', ascending=False)

# Select the top 10 movies from the sorted DataFrame
top_10_movies = movies_list[['title_x', 'vote_count', 'vote_average', 'score']].head(10)


print(top_10_movies)


# In[16]:


popular = movies.sort_values('popularity', ascending=False)

plt.figure(figsize=(12, 4))  # Adjust the figure size
plt.barh(popular['title_x'].head(10), popular['popularity'].head(10), align='center', color='darkblue')
plt.gca().invert_yaxis()
plt.xlabel('popularity')  # Corrected typo
plt.title('Popular Movies')# Corrected typo
plt.ylabel('Movies Title')

plt.show()


# In[17]:




budget = movies.sort_values('budget', ascending=False)
plt.figure(figsize=(12, 4))  # Corrected line
plt.barh(budget['title_x'].head(10), budget['budget'].head(10), align='center', color='lightblue')
plt.gca().invert_yaxis()
plt.xlabel('Budget')
plt.ylabel('Movie Title')
plt.title('Top 10 Movies by Budget')
plt.show()


# In[18]:


movies['overview'].head(10)


# In[19]:



# Now you can proceed with your code
tfdif = TfidfVectorizer(stop_words='english')  # Note the corrected parameter name 'stop_words'

movies['overview'] = movies['overview'].fillna('')  # Replacing empty cells
tfdif_matrix = tfdif.fit_transform(movies['overview'])
print(tfdif_matrix.shape)


# In[20]:


cosine_sim = linear_kernel(tfdif_matrix, tfdif_matrix)


# In[21]:


indices = pd.Series(movies.index, index = movies['title_x']).drop_duplicates()


# In[22]:


def get_recommendations(title,cosine_sim = cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores,key = lambda x:x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    
    movies_indices = [i[0] for i in sim_scores]
    
    return movies['title_x'].iloc[movies_indices]


# In[23]:


get_recommendations('Fight Club')


# In[24]:


print(movies.columns)


# In[25]:


from ast import literal_eval
features = ['cast','crew','genres','keywords']
for feature in features:
    try:
        movies[feature] = movies[feature].apply(literal_eval)
    except ValueError:
        print(f"Error Processing Column'{feature}'")


# In[26]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        return np.nan


# In[27]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        
        if len(names)> 3:
            names = names[:3]
        return names    
    return []        


# In[28]:


movies['director'] =movies['crew'].apply(get_director)

features = ['cast', 'keywords','genres']
for feature in features:
    movies[feature] = movies[feature].apply(get_list)


# In[29]:


movies[['title_x','cast','director','keywords','genres']].head()


# In[30]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace("","")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace("",""))
        else:
            return ''


# In[31]:


features = ['cast','keywords','director','genres']

for feature in features:
    movies[feature] = movies[feature].apply(clean_data)


# In[32]:


def create_soup(x):
    return ' '.join(x['keywords']) +' '+ ' '.join(x['cast'])
movies['soup'] = movies.apply(create_soup, axis = 1)


# In[33]:


count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(movies['soup'])


# In[34]:


cosine_sim2 = cosine_similarity(count_matrix,count_matrix)


# In[35]:


movies = movies.reset_index(drop = True)
indices = pd.Series(movies.index, index= movies['title_x'])


# In[43]:


recommendation = get_recommendations('Toy Story',cosine_sim2)
if recommendation is not None:
    print(recommendation)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




