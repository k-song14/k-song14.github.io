---
layout: post
title: "Generating Lyrics with Genius Lyrics"
authors: Kelly Song, Abhi Vemulapti and Chloe Florit
---

**Hello everyone!**

Today's post will be about demonstrating how to use our Lyric Generator, as well as our building process!

If you'd like to follow along, our code is available in our <a href=https://github.com/k-song14/lyric_generator>Github repository</a>!

We will go through each major portion: web scraping, implementing our Markov model, and integrating web scraping and our model with Flask.

## Web Scraping

In order to be able to generate random lyrics, we need to start with a dataset of lyrics. For this project, we wanted the user to be able to input artists of their choice and generate a song based on those artists' songs. We decided to use a Scrapy spider to scrape the Genius website for lyrics. 

In order to do this, we first created a Scrapy spider (see <a href=https://k-song14.github.io/HW2>here</a> for a tutorial on how to do this). Let's now go step-by-step on our scraping process:

#### Importing Packages

We first imported the necessary packages:

```python

import scrapy
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import requests
import re
from scrapy.crawler import CrawlerProcess

```
We used BeautifulSoup to scrape our lyrics (since Genius has a lot of dynamic content) and CrawlerProcess to be able to run our scraper without using terminal. 

### Creating our Class

Next, we will created our spider class and named it lyrics_spider. We declared an empty list for our start_urls because we will be taking the user input and reformatting it into the generic genius artist page. When the user submits their desired artist(s), our scraper splits them up by comma and reformats them.

```python

class LyricSpider(scrapy.Spider):
    name='lyrics_spider'

    start_urls = []
    
    def __init__(self, form_inp='', **kwargs): # The category variable will have the input 
    form_inp = re.sub(', ', ',', form_inp)
    artists = form_inp.split(',')

    for artist in artists:
        artist = re.sub(' ', '-', artist).capitalize()
        self.start_urls.append('https://genius.com/artists/' + artist + '/')
    print(self.start_urls)
    super().__init__(**kwargs)

```
Finally, it appends the new urls to our start_urls (which contain each artist's Genius page).

### parse method

Once we have our start urls, we will create a soup object to scrape all the links containing "https://genius.com/albums/" on the page. We then get a list of the relative paths for each artist's albums.

```python

    def parse(self,response):
        '''parse method; begins at our initial page 
        goes to our desired next page (artist's albums)
        '''
        #creating our soup object
        page = requests.get(response.url)
        soup = BeautifulSoup(page.content, "html.parser")
        #gets links to all albums on the artist's page
        next_page = [i['href'] for i in soup.find_all("a", href=lambda href: href and "https://genius.com/albums/" in href)]

        #yields request for each url in list
        for link in next_page:
            yield scrapy.Request(link, callback=self.parse_songs)

```

We then iterate over each link in the list and call the next method.

### parse_songs method

Once we're on the album page, we create another soup object that will scrape all the links on the page containing "https://genius.com/" and "lyrics", which will lead us the the song page. However, since the Genius website tends to have many different links on its pages, we want to make sure we only go to the songs by our desired artist(s). To do so, we scrape the page for the artist's name and reformat it so that it matches the generic Genius lyrics page link.

```python

    def parse_songs(self, response):
        '''generates and visits sites of all songs in the album
        '''

        #create soup object
        album_page = requests.get(response.url)
        soup2 = BeautifulSoup(album_page.content, "html.parser")
        #gets all links for songs on the page 
        songs = [i['href'] for i in soup2.find_all("a", href=lambda href: href and "https://genius.com/" and "lyrics" in href)]
        #gets artist name as list
        artist = response.css("a.header_with_cover_art-primary_info-primary_artist::text").get()
        
        #get name inside list
        name = artist[0:]
        #convert to lowercase
        name = name.lower()
        #capitalize only first letter
        name = name[0:].capitalize()
        #replace spaces with hyphen
        name= re.sub(" ", "-", name)

        #yields request for each url in list if it's a song by our desired artist(s)
        for new_link in songs:
            if name not in new_link:
                songs.remove(new_link)
            else:
                yield scrapy.Request(new_link, callback=self.parse_song_page)

```

We add an if statement to ensure that if the artist's name is not present in the link, we remove it from our list. For the remaining songs, we call on the next method.

### parse_song_page method

Finally, we get to scrape our song lyrics. We're able to do so with simple CSS selectors (response.css) and from there, we create a disctionary with the song name and lyrics.

```python

    def parse_song_page(self, response):
        '''creates dictionary with lyrics in each song and corresponding song name
        '''
        #extracts song name from header
        song_name = response.css('span.SongHeaderVariantdesktop__HiddenMask-sc-12tszai-10.bFjDxc::text').get()
        #extracts lyrics from page
        song_lyrics = response.css('span.ReferentFragmentVariantdesktop__Highlight-sc-1837hky-1.jShaMP::text').getall() 

        yield{

        "song": song_name,
        "lyrics": song_lyrics

        }

```

Here is an example of what the scraped lyrics look like. For this specific demonstration, we scraped the lyrics for The Weeknd and Drake and exported it as a csv file (using scrapy crawl lyrics_spider -o lyrics.csv in the terminal), before reading it in as a pandas dataframe (for our webapp, it will be a JSON file and the terminal will not be needed):

```python
import pandas as pd

lyrics = pd.read_csv("lyrics.csv")

lyrics
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
      <th>song</th>
      <th>lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Save Your Tears</td>
      <td>I saw you dancing in a crowded room (Uh),You l...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Call Out My Name</td>
      <td>We found each other,I helped you out of a brok...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Earned It</td>
      <td>I'ma care for you,I'ma care for you, you, you,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Die for You</td>
      <td>I'm findin' ways to articulate the feelin' I'm...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Hills</td>
      <td>Your man on the road, he doin' promo,You said,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>520</th>
      <td>Marvin’s Room</td>
      <td>[Intro: Ericka Lee],Cups of the rosé,The woman...</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Houstatlantavegas</td>
      <td>Hey there, pretty girl,You know exactly what y...</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Headlines</td>
      <td>[Verse 1],I might be too strung out on complim...</td>
    </tr>
    <tr>
      <th>523</th>
      <td>Take Care</td>
      <td>I know you've been hurt by someone else,I can ...</td>
    </tr>
    <tr>
      <th>524</th>
      <td>Shot for Me</td>
      <td>Alisha, Catya, I know that you gon' hear this,...</td>
    </tr>
  </tbody>
</table>
<p>525 rows × 2 columns</p>
</div>

## Markov Model

Once we have our data, we need to run them through a model in order to generate our new song. 

Before we discuss how we implemented our model with our webscraper and flask, we'll go over the initial process of creating our model.

### Model Creation

For the next section of our project, we will tackle predictive text generation using the classic $n$-gram Markov Model which predicts the following character using the transition probability distribution of the preceding $n-1$ characters.

### Generating the Transition Matrix

In our function ```ngram_transition_matrix()```, we construct $n$ long chains of characters using ```consec_chars()```. Then, we construct a multi-indexed dataframe for which each MultiIndex is one of the character chains constructed. We can group by the first $n-1$ indices to obtain the frequency distribution of the following character for each $n-1$ long sequence of characters. From this aggregation we can recover transition probabilities and construct a transition matrix.


```python
from itertools import islice
import re
import numpy as np
import tensorflow as tf
import string

pd.set_option("display.max_rows", None, "display.max_columns", None)

def consec_chars(chars, n):
  '''generates n-long tuples with a sliding window'''
  for i in range(len(chars) - (n-1)):
    yield chars[i:i+n]

def ngram_transition_matrix(chars, n):
  '''Generates a transition matrix given a list of characters'''
  df = pd.DataFrame(consec_chars(chars, n), columns = [str(i) for i in range(1,n+1)])
  
  # for a group of n - 1 preceding words, count the frequency of the following word
  ngroup = list(df)[:n-1]
  counts = df.groupby(ngroup)[str(n)].value_counts()

  # compute relative frequency from counts and 
  # produce a transition matrix using unstack
  probs = (counts / counts.groupby(level=ngroup).sum()).unstack()
  return probs.fillna(0)
```

Next, we remove lowercases and punctuation to standardize the vocabulary.


```python
#Cleaning of text (removes punctuation and makes everything lowercase )

def standardization(text):
    text = text.lower()
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    text = re.sub(r',\s', ',', text)
    return text
```

### Simulation

Finally, we run the simulation and format our output. next_word takes an $n-1$ long sequence and samples the next character, which is then added to the output. Then, we shift our input sequence by one and repeat this process for the desired number of steps, which is decided by the parameters len_verse and len_chorus. The function ```generate_lyrics()``` formats the resulting lyric blocks into a full song, while ```random_start()``` samples an appropriate starting sequences by randomly selecting a character that follows a space.


```python
def next_word(tup, mat):
  '''sample next word given previous n-1 characters'''
  poss = mat.loc[tup,]
  return np.random.choice(list(mat), p = poss)

def generate_block(title, start, length, mat, n):
  '''generates a new text of the given length'''
  start_tup = tuple(start)
  # get current MultiIndex (or regular index if doing bigram)
  curr = start_tup if n > 2 else start
  lyrics = '[' + title + ']' + '\n'
  lyrics += ''.join(start_tup)
  j = 0
  # constructs a sentence for the given seed
  while j < length:
    nxt = next_word(curr, mat)
    lyrics += nxt + ('\n' if nxt == ',' else '')
    # shift window by one to predict next word
    curr = nxt if n == 2 else (*curr[1:], nxt)

    # to ensure the ending of a section does not cut off any words,
    # run the loop until the last character is a space
    if j < length - 1 or curr[-1] == ' ':
      j += 1

  # Corrects spacing at the beginning of lines and in between commas
  lyrics = re.sub(',', ', ', lyrics)
  lyrics = re.sub(' ,', ',', lyrics)
  return re.sub('\n\s', '\n', lyrics) + '\n\n'

def random_start(text, n):
  '''generate the starting word for a song section'''
  spaces = [i for i in range(len(text)) if text[i] == ' ']
  j = np.random.choice(spaces)
  return text[j+1:j+n]

def generate_song(text, n=2, len_verse=500, len_chorus=100, num_verses=2):
  '''simulate the Markov chain and format appropriately'''
  np.random.seed()

  # create random starting phrases
  text = standardization(text)
  starts = [random_start(text, n) for i in range(num_verses+1)] 
  chars = list(text)

  mat = ngram_transition_matrix(chars, n)
  
  song = ""
  chorus = generate_block('Chorus', starts[0], len_chorus, mat, n)

  for i in range(num_verses):
    song += generate_block('Verse ' + str(i+1), starts[i+1], len_verse, mat, n)
    song += chorus

  return song

```

Given that our webapp will scrape the lyrics data into a JSON, we clean up the data as follows and concatenate into a single string that is used as the model input.


```python
lyrics_df = pd.read_json(r'./file.json')
lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(lambda l: ', '.join(l))
lyrics_input = ' '.join(lyrics_df['lyrics'].apply(lambda t: str(t)))
```

We would also be interested in looking at some visualizations of the data. We plot the transition probabilities for the phrases "you know " and "i ". Note that the two phrases generate very different distributions because the latter phrase leaves more possibilities.


```python
import plotly.express as px

def transition_dist_plot(phrase):
  text = standardization(lyrics_input)
  chars = list(text)
  dist_df = ngram_transition_matrix(chars, len(phrase) + 1).loc[tuple(phrase),:].reset_index()
  df = pd.DataFrame()
  df['char'] = dist_df.iloc[:,0]
  df['prob'] = dist_df.iloc[:,1]
  
  fig = px.histogram(df, x='char', y='prob', title="Transition Probabilities for Starting Phrase '" + phrase + "'")
  fig.write_html(phrase + 'dist.html')
  return fig

transition_dist_plot('you know ').show()
```

<div class = "display">
{% include transition_dist_i.html %}
</div>

Finally, let us run the code on the example dataset with lyrics from Taylor Swift.


```python
print(generate_song(lyrics_input, n=10))
```

    [Verse 1]
    i drive away with you i'd dance, 
    in a storm in my best dress, 
    fearless so why don't you, 
    don't you, 
    don't you think about is karma, 
    and then i heard you move to me like, 
    you could write my name is up in lights, 
    but i kept you like i want to, 
    everything, 
    i hold onto the night we could leave the christmas lights go out, 
    it's hard not to fall in love with a careless man's careful daughter, 
    you are in love 'til it hurts or bleeds, 
    or fades in time, 
    'cause it could've been me, 
    for digging up the phone as you whisper in the 
    
    [Chorus]
    show you incredible things?", 
    like, 
    "oh, 
    my, 
    what a marvelous time ruining everything, 
    a marvelous time ruining everything 
    
    [Verse 2]
    mother's eyes, 
    his father's ambition, 
    i'll bright to your face in an invisible seems the only one who’s got enough, 
    i get drunk, 
    but it's best if we both cried, 
    and when i felt like i was again tonight, 
    , 
    'cause there we are again when nobody shine through the sleepless night, 
    the snaps from the city, 
    and i hope you know it used to be mad anymore, 
    and you know i love the drought was the moment i knew, 
    oh-oh-oh-oh-oh-oh-oh-oh, 
    you never met, 
    but loving him was red, 
    yeah, 
    yeah, 
    and it's really brought it, 
    champagne problems, 
    your 
    
    [Chorus]
    show you incredible things?", 
    like, 
    "oh, 
    my, 
    what a marvelous time ruining everything, 
    a marvelous time ruining everything 
    
    
 Now that we have our model and our data, our last step is to integrate them into a webapp.
 
 ## Flask WebApp
 
 ### Integrating Scrapy and our Markov Model
 
We created a web app using flask. The first page of our web app is home page containing an "about" section, "lyric generator" section where the user input artist names, and a "bio" section with creator photos. First we scraped lyrics from flask based on the user's input. This outputs a json file, which we will run through our markov model in flask. The markov model generates lyrics which we will print on the "scrape" page of our web app. 


```python
import crochet
crochet.setup()
import pandas as pd
import json
from flask import Flask , render_template, jsonify, request, redirect, url_for
import scrapy
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import requests
import re
from scrapy.crawler import CrawlerRunner
from scrapy import signals
from scrapy.signalmanager import dispatcher
import time
from itertools import islice
import re
import numpy as np
import string
import os
# Importing our Scraping Function from the lyrics file
from lyrics1.lyrics1.spiders.lyrics import ImdbSpider

### stuff from last class
app = Flask(__name__)

output_data = []
crawl_runner = CrawlerRunner()

# By Deafult Flask will come into this when we run the file
@app.route('/')
def index():
	return render_template("index.html") # Returns index.html file in templates folder.


# After clicking the Submit Button FLASK will come into this
@app.route('/', methods=['POST'])
def submit():
    if request.method == 'POST':
        s = request.form['inp'] # Getting the Input 
        global inp
        inp = s
        
    # This will remove any existing file with the same name so that the scrapy will not append the data to any previous file.
    if os.path.exists("./webApp/app/file.json"): 
        os.remove("./webApp/app/file.json")

    return redirect(url_for('scrape')) # Passing to the Scrape function


@app.route("/scrape")
def scrape():

    scrape_with_crochet(form_inp=inp) # Passing that URL to our Scraping Function

    time.sleep(100)
    with open('file.json', 'w') as f:
        json.dump(output_data, f)
    
    lyrics_df = pd.read_json('./file.json')
    lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(lambda l: ', '.join(l))

    lyrics_input = ' '.join(lyrics_df['lyrics'].apply(lambda t: str(t)))
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    def consec_chars(chars, n):
        '''generates n-long tuples with a sliding window'''
        for i in range(len(chars) - (n-1)):
            yield chars[i:i+n]
    
    def ngram_transition_matrix(chars, n):
        '''Generates a transition matrix given a list of characters'''
        df = pd.DataFrame(consec_chars(chars, n), columns = [str(i) for i in range(1,n+1)])
        ngroup = list(df)[:n-1]
        counts = df.groupby(ngroup)[str(n)].value_counts()
        probs = (counts / counts.groupby(level=ngroup).sum()).unstack()
        return probs.fillna(0)
    
    def standardization(text):
        text = text.lower()
        text = re.sub(r'[\(\[].*?[\)\]]', '', text)
        text = re.sub(r',\s', ',', text)
        return text

    def next_word(tup, mat):
        '''sample next word given previous n-1 characters'''
        poss = mat.loc[tup,]
        return np.random.choice(list(mat), p = poss)

    def generate_block(title, start, length, mat, n):
        '''generates a new text of the given length'''
        start_tup = tuple(start)
        # get current MultiIndex (or regular index if doing bigram)
        curr = start_tup if n > 2 else start
        lyrics = '[' + title + ']' + '\n'
        lyrics += ''.join(start_tup)
        j = 0
        while j < length:
            nxt = next_word(curr, mat)
            lyrics += nxt + ('\n' if nxt == ',' else '')
            curr = nxt if n == 2 else (*curr[1:], nxt)
            if j < length - 1 or curr[-1] == ' ':
                j += 1
        
        lyrics = re.sub(',', ', ', lyrics)
        lyrics = re.sub(' ,', ',', lyrics)
        return re.sub('\n\s', '\n', lyrics) + '\n\n'
    
    def random_start(text, n):
        spaces = [i for i in range(len(text)) if text[i] == ' ']
        j = np.random.choice(spaces)
        return text[j+1:j+n]
    
    def generate_song(text, n=2, len_verse=500, len_chorus=100, num_verses=2):
        np.random.seed()
        text = standardization(text)
        starts = [random_start(text, n) for i in range(num_verses+1)]
        chars = list(text)
        mat = ngram_transition_matrix(chars, n)
        
        song = ""
        chorus = generate_block('Chorus', starts[0], len_chorus, mat, n)

        for i in range(num_verses):
            song += generate_block('Verse ' + str(i+1), starts[i+1], len_verse, mat, n)
            song += chorus
        
        return song
    
    word = generate_song(lyrics_input, n=10)

    return render_template('scrape.html', word = word) # Returns the scraped data after being running for 20 seconds.

@crochet.run_in_reactor
def scrape_with_crochet(form_inp):
    # This will connect to the dispatcher that will kind of loop the code between these two functions.
    dispatcher.connect(_crawler_result, signal=signals.item_scraped)
    
    # This will connect to the ReviewspiderSpider function in our scrapy file and after each yield will pass to the crawler_result function.
    eventual = crawl_runner.crawl(ImdbSpider, form_inp = form_inp)
    return eventual

#This will append the data to the output data list.
def _crawler_result(item, response, spider):
    output_data.append(dict(item))


if __name__== "__main__":
    app.run(debug=True)
```

And our webapp should work after running it in the terminal! 

Let's look at some examples. For our first example, we just used Taylor Swift, and for our second, we used Taylor Swift, Katy Perry and Dua Lipa (aka redeeming ourselves from our presentation after we readded the code we accidentally deleted): 

#### Example 1: Taylor Swift

![tswift1.jpeg](/images/tswift1.jpeg)
![tswift2.jpeg](/images/tswift2.jpeg)

### Example 2: Taylor Swift, Katy Perry, Dua Lipa

![3artist1.png](/images/3artist1.png)
![3artist2.png](/images/3artist2.png)

### Designing our Website

Now that we have our web app up and running, we can move onto designing the website using HTML and CSS! For our webapp, we used two different style sheets: one provided by w3.css (so we can just call on the different classes) and mystyle.css, which we used to fix spacing, font, and color. Our idea was to have our web app resemble the Genius website.

Here are some gifs showcasing our website and its features, as well as the song generator!

##### Website Features
![lyricgif](/images/ezgif.com-video-to-gif.gif)
![lyricgif](/images/ezgif.com-video-to-gif-3.gif)

#### Generated Song
![lyricgif2](/images/ezgif.com-video-to-gif-2.gif)

Thank you so much for reading!
