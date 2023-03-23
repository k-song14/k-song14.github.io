# DH 150 Project Sentiment Analysis

## Imports


```python
# import necessary packages

import re 

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob 

from wordcloud import WordCloud
```


```python
# import data

twitter_data = pd.read_csv("Twitter Takeover Data - takeover_tweets_clean.csv")
```


```python
# check data

twitter_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>renderedContent</th>
      <th>replyCount</th>
      <th>retweetCount</th>
      <th>likeCount</th>
      <th>user</th>
      <th>time</th>
      <th>followersCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/26/22</td>
      <td>‘Chief Twit’ Elon Musk visits Twitter headquar...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>BusinessTimes</td>
      <td>23:59:00</td>
      <td>59349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/26/22</td>
      <td>Twitter employees leave company in droves ahea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IWZen</td>
      <td>23:58:00</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/26/22</td>
      <td>Just so you know, even if Elon Musk does succe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>AltLeloge</td>
      <td>23:58:00</td>
      <td>2793</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/26/22</td>
      <td>@elonmusk The ‘Woke’ Twitter Meltdown Has Begu...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>manolo9927</td>
      <td>23:56:00</td>
      <td>1661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/26/22</td>
      <td>@Twitter The ‘Woke’ Twitter Meltdown Has Begun...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>manolo9927</td>
      <td>23:55:00</td>
      <td>1661</td>
    </tr>
  </tbody>
</table>
</div>



## Text Prepreocessing


```python
# text cleaning, tokenization and lemmatization

stop_words = stopwords.words('english')
# WordNetLemmatizer object
wnl = WordNetLemmatizer()

def process_tweet(tweet):
    # standardize tweet
    lowercase = tweet.lower() # lowercase
    r = re.sub(r'[^\w\s]','', lowercase) # remove punctuation
    r = re.sub(r'http\S+', '', r) # remove links
    r = re.sub("@[A-Za-z0-9_]+","", r) # remove usernames
    r = re.sub("#[A-Za-z0-9_]+","", r) # remove hashtags
    r = re.sub('[()!?]', ' ', r) # remove brackets, parentheses, etc.
    
    # tokenize tweet
    
    tokenize = word_tokenize(r)
    token = [i for i in tokenize if i not in stop_words]
    
    # lemmatize tweet
    
    lemm = [wnl.lemmatize(words) for words in token]
    
    # turn back into sentence
    
    final_tweet = ' '.join(lemm)
    
    return final_tweet
```


```python
# turn tweets column into list

tweet_list = twitter_data["renderedContent"].to_list()
```


```python
cleaned_tweets = [process_tweet(i) for i in tweet_list]
```


```python
cleaned_tweets[0]
```




    'chief twit elon musk visit twitter headquarters takeover deadline loom btsgwlw3'



## Sentiment Analysis


```python
# begin sentiment analysis

# polarity scores and tweet text

sentiment_obj = [TextBlob(tweet) for tweet in cleaned_tweets]
scores = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_obj]

# check

scores[0]
```




    [0.0,
     'chief twit elon musk visit twitter headquarters takeover deadline loom btsgwlw3']




```python
# make dataframe of tweets and corresponding polarity scores

sentiment_df = pd.DataFrame(scores, columns=["polarity", "tweet"])

sentiment_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>polarity</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>chief twit elon musk visit twitter headquarter...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>twitter employee leave company drove ahead elo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>know even elon musk succeed taking twitter lea...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.136364</td>
      <td>elonmusk woke twitter meltdown begun new repor...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.136364</td>
      <td>twitter woke twitter meltdown begun new report...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45075</th>
      <td>0.000000</td>
      <td>socalled poison pill twitter proposed use elon...</td>
    </tr>
    <tr>
      <th>45076</th>
      <td>0.000000</td>
      <td>mpukita jack elonmusk board member twitter hel...</td>
    </tr>
    <tr>
      <th>45077</th>
      <td>0.300000</td>
      <td>twitter need answer john question imposing rul...</td>
    </tr>
    <tr>
      <th>45078</th>
      <td>-0.100000</td>
      <td>elon musk said sec filing exploring whether la...</td>
    </tr>
    <tr>
      <th>45079</th>
      <td>0.000000</td>
      <td>point hear one 20 podcasts listen talk elonmus...</td>
    </tr>
  </tbody>
</table>
<p>45080 rows × 2 columns</p>
</div>




```python
# make new column for sentiments

sentiment_df['sentiments'] = sentiment_df['polarity']


# for loop for classification
for i in range(len(sentiment_df)):
    if sentiment_df['polarity'][i]>0:
        sentiment_df['sentiments'][i] = "Positive"
    elif sentiment_df['polarity'][i]<0:
        sentiment_df['sentiments'][i] = "Negative"
    else:
        sentiment_df['sentiments'][i] = "Neutral"
```

    /Users/kellysong/opt/anaconda3/envs/PIC16B/lib/python3.7/site-packages/ipykernel_launcher.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      del sys.path[0]
    /Users/kellysong/opt/anaconda3/envs/PIC16B/lib/python3.7/site-packages/pandas/core/indexing.py:1732: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_single_block(indexer, value, name)



```python
sentiment_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>polarity</th>
      <th>tweet</th>
      <th>sentiments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>chief twit elon musk visit twitter headquarter...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>twitter employee leave company drove ahead elo...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>know even elon musk succeed taking twitter lea...</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get count of each sentiment

sentiment_df.groupby('sentiments')['tweet'].count(),
```




    (sentiments
     Negative     7984
     Neutral     22010
     Positive    15086
     Name: tweet, dtype: int64,)



## Data Visualizations


```python
# Pie chart

plt.pie(sentiment_df.groupby('sentiments')['tweet'].count(), labels=["Negative", "Neutral", "Positive"], autopct='%1.1f%%')
plt.title("Percentage of Positive, Negative and Neutral Sentiments")
```




    Text(0.5, 1.0, 'Percentage of Positive, Negative and Neutral Sentiments')




    
![png](output_17_1.png)
    



```python
# word cloud

words = ' '.join([text for text in cleaned_tweets])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
```


    
![png](output_18_0.png)
    


## Final Dataframe 


```python
sentiment_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>polarity</th>
      <th>tweet</th>
      <th>sentiments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>chief twit elon musk visit twitter headquarter...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>twitter employee leave company drove ahead elo...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>know even elon musk succeed taking twitter lea...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.136364</td>
      <td>elonmusk woke twitter meltdown begun new repor...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.136364</td>
      <td>twitter woke twitter meltdown begun new report...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45075</th>
      <td>0.000000</td>
      <td>socalled poison pill twitter proposed use elon...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>45076</th>
      <td>0.000000</td>
      <td>mpukita jack elonmusk board member twitter hel...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>45077</th>
      <td>0.300000</td>
      <td>twitter need answer john question imposing rul...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>45078</th>
      <td>-0.100000</td>
      <td>elon musk said sec filing exploring whether la...</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>45079</th>
      <td>0.000000</td>
      <td>point hear one 20 podcasts listen talk elonmus...</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
<p>45080 rows × 3 columns</p>
</div>




```python
my_cols = ["date", "time","replyCount", "retweetCount", "likeCount", "user", "followersCount"]

for i in my_cols:
    sentiment_df.insert(1, i, twitter_data[i])
```


```python
sentiment_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>polarity</th>
      <th>followersCount</th>
      <th>user</th>
      <th>likeCount</th>
      <th>retweetCount</th>
      <th>replyCount</th>
      <th>time</th>
      <th>date</th>
      <th>tweet</th>
      <th>sentiments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>59349</td>
      <td>BusinessTimes</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>23:59:00</td>
      <td>10/26/22</td>
      <td>chief twit elon musk visit twitter headquarter...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>107</td>
      <td>IWZen</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23:58:00</td>
      <td>10/26/22</td>
      <td>twitter employee leave company drove ahead elo...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>2793</td>
      <td>AltLeloge</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>23:58:00</td>
      <td>10/26/22</td>
      <td>know even elon musk succeed taking twitter lea...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.136364</td>
      <td>1661</td>
      <td>manolo9927</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>23:56:00</td>
      <td>10/26/22</td>
      <td>elonmusk woke twitter meltdown begun new repor...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.136364</td>
      <td>1661</td>
      <td>manolo9927</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23:55:00</td>
      <td>10/26/22</td>
      <td>twitter woke twitter meltdown begun new report...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45075</th>
      <td>0.000000</td>
      <td>1935455</td>
      <td>ntvuganda</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>5:57:00</td>
      <td>4/22/22</td>
      <td>socalled poison pill twitter proposed use elon...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>45076</th>
      <td>0.000000</td>
      <td>182</td>
      <td>ClegTreat</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5:52:00</td>
      <td>4/22/22</td>
      <td>mpukita jack elonmusk board member twitter hel...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>45077</th>
      <td>0.300000</td>
      <td>312</td>
      <td>LogosMMXX</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5:50:00</td>
      <td>4/22/22</td>
      <td>twitter need answer john question imposing rul...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>45078</th>
      <td>-0.100000</td>
      <td>127472</td>
      <td>dealbook</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5:46:00</td>
      <td>4/22/22</td>
      <td>elon musk said sec filing exploring whether la...</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>45079</th>
      <td>0.000000</td>
      <td>68</td>
      <td>TD_Dreizehn</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5:44:00</td>
      <td>4/22/22</td>
      <td>point hear one 20 podcasts listen talk elonmus...</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
<p>45080 rows × 10 columns</p>
</div>




```python
twitter_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>renderedContent</th>
      <th>replyCount</th>
      <th>retweetCount</th>
      <th>likeCount</th>
      <th>user</th>
      <th>time</th>
      <th>followersCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/26/22</td>
      <td>‘Chief Twit’ Elon Musk visits Twitter headquar...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>BusinessTimes</td>
      <td>23:59:00</td>
      <td>59349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/26/22</td>
      <td>Twitter employees leave company in droves ahea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IWZen</td>
      <td>23:58:00</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/26/22</td>
      <td>Just so you know, even if Elon Musk does succe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>AltLeloge</td>
      <td>23:58:00</td>
      <td>2793</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/26/22</td>
      <td>@elonmusk The ‘Woke’ Twitter Meltdown Has Begu...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>manolo9927</td>
      <td>23:56:00</td>
      <td>1661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/26/22</td>
      <td>@Twitter The ‘Woke’ Twitter Meltdown Has Begun...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>manolo9927</td>
      <td>23:55:00</td>
      <td>1661</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45075</th>
      <td>4/22/22</td>
      <td>The so-called "poison pill" Twitter has propos...</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>ntvuganda</td>
      <td>5:57:00</td>
      <td>1935455</td>
    </tr>
    <tr>
      <th>45076</th>
      <td>4/22/22</td>
      <td>@mpukita @jack @elonmusk A board member of Twi...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>ClegTreat</td>
      <td>5:52:00</td>
      <td>182</td>
    </tr>
    <tr>
      <th>45077</th>
      <td>4/22/22</td>
      <td>@Twitter need an answer to John's question... ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>LogosMMXX</td>
      <td>5:50:00</td>
      <td>312</td>
    </tr>
    <tr>
      <th>45078</th>
      <td>4/22/22</td>
      <td>Elon Musk said in an SEC filing that he was ex...</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>dealbook</td>
      <td>5:46:00</td>
      <td>127472</td>
    </tr>
    <tr>
      <th>45079</th>
      <td>4/22/22</td>
      <td>At this point if I hear one more of the 20+ po...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>TD_Dreizehn</td>
      <td>5:44:00</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
<p>45080 rows × 8 columns</p>
</div>



## Final Dataframe #2 (FINAL!!)


```python
twitter_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>renderedContent</th>
      <th>replyCount</th>
      <th>retweetCount</th>
      <th>likeCount</th>
      <th>user</th>
      <th>time</th>
      <th>followersCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/26/22</td>
      <td>‘Chief Twit’ Elon Musk visits Twitter headquar...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>BusinessTimes</td>
      <td>23:59:00</td>
      <td>59349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/26/22</td>
      <td>Twitter employees leave company in droves ahea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IWZen</td>
      <td>23:58:00</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/26/22</td>
      <td>Just so you know, even if Elon Musk does succe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>AltLeloge</td>
      <td>23:58:00</td>
      <td>2793</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/26/22</td>
      <td>@elonmusk The ‘Woke’ Twitter Meltdown Has Begu...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>manolo9927</td>
      <td>23:56:00</td>
      <td>1661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/26/22</td>
      <td>@Twitter The ‘Woke’ Twitter Meltdown Has Begun...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>manolo9927</td>
      <td>23:55:00</td>
      <td>1661</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ["sentiments", "polarity"]

for i in cols:
    twitter_data.insert(1, i, sentiment_df[i])
```


```python
twitter_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>polarity</th>
      <th>sentiments</th>
      <th>renderedContent</th>
      <th>replyCount</th>
      <th>retweetCount</th>
      <th>likeCount</th>
      <th>user</th>
      <th>time</th>
      <th>followersCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/26/22</td>
      <td>0.000000</td>
      <td>Neutral</td>
      <td>‘Chief Twit’ Elon Musk visits Twitter headquar...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>BusinessTimes</td>
      <td>23:59:00</td>
      <td>59349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/26/22</td>
      <td>0.000000</td>
      <td>Neutral</td>
      <td>Twitter employees leave company in droves ahea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IWZen</td>
      <td>23:58:00</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/26/22</td>
      <td>0.000000</td>
      <td>Neutral</td>
      <td>Just so you know, even if Elon Musk does succe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>AltLeloge</td>
      <td>23:58:00</td>
      <td>2793</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/26/22</td>
      <td>0.136364</td>
      <td>Positive</td>
      <td>@elonmusk The ‘Woke’ Twitter Meltdown Has Begu...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>manolo9927</td>
      <td>23:56:00</td>
      <td>1661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/26/22</td>
      <td>0.136364</td>
      <td>Positive</td>
      <td>@Twitter The ‘Woke’ Twitter Meltdown Has Begun...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>manolo9927</td>
      <td>23:55:00</td>
      <td>1661</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
