#!/usr/bin/env python
# coding: utf-8

# # 영화 추천 시스템

# 1. Demographic Filtering (인구통계학적 필터링)
# 1. content Based Filtering (컨텐츠 기반 필터링)
# 1. Collaborative Filtering (협업 필터링) 

# ## 1.Demographic Filtering (인구통계학적 필터링)

# In[1]:


import pandas as pd
import numpy as np

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')


# In[2]:


df1.head()


# In[3]:


df2.head(3)


# In[4]:


df1.shape, df2.shape


# In[5]:


df1['title'].equals(df2['title'])


# In[6]:


df1.columns


# In[7]:


df1.columns = ['id', 'title', 'cast', 'crew']
df1.columns


# In[8]:


df1[['id', 'cast', 'crew']]


# In[9]:


df2 = df2.merge(df1[['id', 'cast', 'crew']], on='id')
df2.head(3)


# 영화 1 : 영화의 평점이 10/10 -> 5명이 평가  
# 영화 2 : 영화의 평점이 8/10 -> 500명이 평가

# In[10]:


C = df2['vote_average'].mean()
C


# In[11]:


m = df2['vote_count'].quantile(0.9)
m


# In[12]:


q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape


# In[13]:


q_movies['vote_count'].sort_values()


# In[14]:


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


# In[15]:


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies.head(3)


# In[16]:


q_movies = q_movies.sort_values('score', ascending=False)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


# In[17]:


pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# ## 2. Content Based Filtering (컨텐츠 기반 필터링)

# ### 줄거리 기반 추천

# In[18]:


df2['overview'].head(5)


# Bag Of Words - BOW

# 문장1 : I am a boy
# 
# 문장2 : I am a girl
# 
# I(2), am(2), a(2), boy(1), girl(1)
# 
#       I     am     a     boy    girl
# 문장1 1     1      1      1      0 (1,1,1,0)
#  (I am a boy)
#  
# 문장2 1     1      1      0      1 (1,1,0,1)
#  (I am a girl)
#  
#  피처 벡터화.
#  
#  문서 100개
#  모든 문서에서 나온 단어 10,000개
#  100* 10,000 - 100만
#  
#          단어1, 단어2, 단어3, 단어4, ...... 단어 10000
# 문서1      1      1      3      0
# 문서2
# 문서3
# ..
# 문서100
# 
# 1. TfidfVectorizer (TF-IDF 기반의 벡터화)
# 2. CountVectorizer

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')


# In[20]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ENGLISH_STOP_WORDS


# In[21]:


df2['overview'].isnull().values.any()


# In[22]:


df2['overview'] = df2['overview'].fillna('')


# In[23]:


tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape


# In[24]:


tfidf_matrix


# In[25]:


from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim


# | | 문장1 | 문장2 | 문장3 |
# |---|---|---|---|
# |문장1|1|0.3|0.8|
# |문장2|0.3|1|0.5|
# |문장3|0.8|0.5|1|

# In[26]:


cosine_sim.shape


# In[27]:


indices = pd.Series(df2.index, index=df2['title']). drop_duplicates()
indices


# In[28]:


indices['The Dark Knight Rises']


# In[29]:


df2.iloc[[3]]


# In[30]:


# 영화의 제목을 입력받으면 코사인 유사도를 통해서 가장 유사도가 높은 상위 10개의 영화 목록 반환
def get_recommendations(title, cosine_sim=cosine_sim):
     # 영화 제목을 통해서 전체 테이터 기준 그 영화의 index 값을 얻기
    idx = indices[title]
    # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]
    
    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices = [i[0] for i in sim_scores]
                     
    # 인덱스 정보를 통해 영화 제목 추출
    return df2['title'].iloc[movie_indices]              


# In[31]:


test_idx = indices['The Dark Knight Rises'] # 영화 제목을 통해서 전체 테이터 기준 그 영화의 index 값을 얻기
test_idx


# In[32]:


cosine_sim[3]


# In[33]:


test_sim_scores = list(enumerate(cosine_sim[3])) # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기


# In[34]:


test_sim_scores = sorted(test_sim_scores, key=lambda x: x[1], reverse=True) # 코사인 유사도 기준으로 내림차순 정렬
test_sim_scores[1:11] # 자기 자신을 제외한 10ㄱ의 추천 영화를 슬라이싱


# In[35]:


def get_second(x):
    return x[1]

lst = ['인덱스', '유사도']
print(get_second(lst))


# In[36]:


(lambda x: x[1])(lst)


# In[37]:


# 추천 영화 목록 10개의 인덱스 정보 추출
test_movie_indices = [i[0] for i in test_sim_scores[1:11]]
test_movie_indices


# In[38]:


# 인덱스 정보를 통해 영화 제목 추출
df2['title'].iloc[test_movie_indices]


# In[39]:


df2['title'][:20]


# In[40]:


get_recommendations('Avengers: Age of Ultron')


# In[41]:


get_recommendations('The Avengers')


# In[42]:


get_recommendations('The Dark Knight Rises')


# ### 다양한 요소 기반 추천 (장르, 감독, 키워드 등)

# In[43]:


df2.head(3)


# In[44]:


df2.loc[0, 'genres']


# In[45]:


s1 = [{"id": 28, "name": "Action"}]
s2 = '[{"id": 28, "name": "Action"}]'


# In[46]:


type(s1), type(s2)


# In[47]:


from ast import literal_eval
s2 = literal_eval(s2)
s2, type(s2)


# In[48]:


print(s1)
print(s2)


# In[49]:


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


# In[50]:


df2.loc[0, 'crew']


# In[51]:


# 감독 정보를 추출
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[52]:


df2['director'] = df2['crew'].apply(get_director)
df2['director']


# In[53]:


df2[df2['director'].isnull()]


# In[54]:


df2.loc[0, 'cast']


# In[55]:


df2.loc[0, 'genres']


# In[56]:


df2.loc[0, 'keywords']


# In[57]:


# 처음 3개의 데이터 중에서 name 에 해당하는 value 만 추출
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# In[58]:


features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)


# In[59]:


df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[60]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''


# In[61]:


features = ['cast', 'keywords', 'genres', 'director']
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# In[62]:


df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[63]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)
df2['soup']


# In[64]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
count_matrix


# In[65]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
cosine_sim2


# In[66]:


indices['Avatar']


# In[67]:


df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
indices


# In[68]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# In[69]:


get_recommendations('Up', cosine_sim2)


# In[70]:


get_recommendations('The Martian', cosine_sim2)


# In[71]:


indices['The Martian']


# In[72]:


df2.loc[270]


# In[73]:


df2.loc[4]


# In[74]:


get_recommendations('The Avengers', cosine_sim2)


# In[75]:


import pickle


# In[76]:


df2.head(3)


# In[77]:


movies = df2[['id', 'title']].copy()
movies.head(5)


# In[78]:


pickle.dump(movies, open('movies.pickle', 'wb'))


# In[79]:


pickle.dump(cosine_sim2, open('cosine_sim.pickle', 'wb'))

