# Importing necessary packages and data



```python
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from tensorflow.keras import layers
from tensorflow.keras import losses

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.model_selection import train_test_split
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"

train_data = pd.read_csv(train_url)
train_data
```





  <div id="df-cbee4a89-4ec8-468c-91ff-b0bfbda9174e">
    <div class="colab-df-container">
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
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22444</th>
      <td>10709</td>
      <td>ALARMING: NSA Refuses to Release Clinton-Lynch...</td>
      <td>If Clinton and Lynch just talked about grandki...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22445</th>
      <td>8731</td>
      <td>Can Pence's vow not to sling mud survive a Tru...</td>
      <td>() - In 1990, during a close and bitter congre...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22446</th>
      <td>4733</td>
      <td>Watch Trump Campaign Try To Spin Their Way Ou...</td>
      <td>A new ad by the Hillary Clinton SuperPac Prior...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22447</th>
      <td>3993</td>
      <td>Trump celebrates first 100 days as president, ...</td>
      <td>HARRISBURG, Pa.U.S. President Donald Trump hit...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22448</th>
      <td>12896</td>
      <td>TRUMP SUPPORTERS REACT TO DEBATE: “Clinton New...</td>
      <td>MELBOURNE, FL is a town with a population of 7...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>22449 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cbee4a89-4ec8-468c-91ff-b0bfbda9174e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cbee4a89-4ec8-468c-91ff-b0bfbda9174e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cbee4a89-4ec8-468c-91ff-b0bfbda9174e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Make function to clean data


```python
def make_dataset(df):
  # Remove stopwords from the article text and title. 
  # A stopword is a word that is usually considered to be uninformative, such as “the,” “and,” or “but.” 
  # Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form (title, text), 
  # and the output should consist only of the fake column. You may find it helpful to consult lecture notes or this tutorial for 
  # reference on how to construct and use Datasets with multiple inputs

  #stop words
  stop = stopwords.words('english')

  #remove stop words from article title
  df["title"] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  #remove stop words from article text
  df["text"] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

  data = tf.data.Dataset.from_tensor_slices(
   ( # dictionary for input data/features
    #our two inputs: title and text
    { "title": df[["title"]],
     "text": df["text"]
    },
    # dictionary for output data/labels
    # one output: fake
    { "fake": df[["fake"]]
        
    }   
   ) 
  ) 

  return data.batch(100)

```


```python
#batch our data to make training faster; train on chunks of data rather than individual rows
data = make_dataset(train_data)
```


```python
#check size of dataset
len(data)
```




    225



# Split data into training and validation sets


```python
#shuffle dataset
import random
random.seed(10)
data = data.shuffle(buffer_size = len(data))

#20% validation
val_size  = int(0.2*len(data))

val = data.take(val_size)
train = data.skip(val_size).take(len(data) - val_size)

#check size of training and validation sets
print(len(train), len(val))
```

    180 45


# Base rate - Labels Iterator, fake text, count of labels, on training data


```python
# Base rate 
## similar to previous hw; can get true and fake from training data, labels iterator on fake column

labels_iterator= train.unbatch().map(lambda input, output: output).as_numpy_iterator()

train_data2 = train_data.sample(n=1800)

len(train_data2[train_data2["fake"] == 1]) / 1800
```




    0.5355555555555556



The base rate appears to be about 53.6%, which indicates that a little more than half of the artiles in the training data is fake while the other half is true.

# Model Creation


```python
# Text vectorization

#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

## Title Vectorization
title_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

title_vectorize_layer.adapt(train.map(lambda x, y: x["title"]))

## Text Vectorization

text_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, 
    output_mode='int',
    output_sequence_length=500) 

text_vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```

### Model 1: Title only


```python
title_input = tf.keras.Input(
    shape=(1,),
    name = "title", # same name as the dictionary key in the dataset
    dtype = "string"
)
```


```python
# layers for processing the title
title_features = title_vectorize_layer(title_input)

# Add embedding layer, dropout
title_features = layers.Embedding(size_vocabulary, output_dim = 3, name="embedding1")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(2, activation='relu')(title_features)

output = layers.Dense(2, name="fake")(title_features) 
```


```python
model1 = tf.keras.Model(
    inputs = title_input,
    outputs = output
)
```


```python
from tensorflow.keras import utils

model1.summary()
utils.plot_model(model1)
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization_2 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding1 (Embedding)      (None, 500, 3)            6000      
                                                                     
     dropout_2 (Dropout)         (None, 500, 3)            0         
                                                                     
     global_average_pooling1d_1   (None, 3)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_3 (Dropout)         (None, 3)                 0         
                                                                     
     dense_1 (Dense)             (None, 2)                 8         
                                                                     
     fake (Dense)                (None, 2)                 6         
                                                                     
    =================================================================
    Total params: 6,014
    Trainable params: 6,014
    Non-trainable params: 0
    _________________________________________________________________





    
![png](output_18_1.png)
    




```python
model1.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history = model1.fit(train, validation_data=val, epochs=20)
```

    Epoch 1/20


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning: Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)


    180/180 [==============================] - 3s 11ms/step - loss: 0.6922 - accuracy: 0.5199 - val_loss: 0.6909 - val_accuracy: 0.5244
    Epoch 2/20
    180/180 [==============================] - 2s 9ms/step - loss: 0.6889 - accuracy: 0.5577 - val_loss: 0.6845 - val_accuracy: 0.5988
    Epoch 3/20
    180/180 [==============================] - 2s 9ms/step - loss: 0.6764 - accuracy: 0.6732 - val_loss: 0.6650 - val_accuracy: 0.8622
    Epoch 4/20
    180/180 [==============================] - 2s 9ms/step - loss: 0.6485 - accuracy: 0.8005 - val_loss: 0.6267 - val_accuracy: 0.9313
    Epoch 5/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.6015 - accuracy: 0.8615 - val_loss: 0.5699 - val_accuracy: 0.9285
    Epoch 6/20
    180/180 [==============================] - 2s 9ms/step - loss: 0.5442 - accuracy: 0.8794 - val_loss: 0.5050 - val_accuracy: 0.9322
    Epoch 7/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.4829 - accuracy: 0.8984 - val_loss: 0.4426 - val_accuracy: 0.9338
    Epoch 8/20
    180/180 [==============================] - 2s 9ms/step - loss: 0.4248 - accuracy: 0.9200 - val_loss: 0.3848 - val_accuracy: 0.9338
    Epoch 9/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.3756 - accuracy: 0.9275 - val_loss: 0.3347 - val_accuracy: 0.9520
    Epoch 10/20
    180/180 [==============================] - 2s 13ms/step - loss: 0.3306 - accuracy: 0.9383 - val_loss: 0.2911 - val_accuracy: 0.9531
    Epoch 11/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.2931 - accuracy: 0.9438 - val_loss: 0.2575 - val_accuracy: 0.9642
    Epoch 12/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.2606 - accuracy: 0.9494 - val_loss: 0.2229 - val_accuracy: 0.9651
    Epoch 13/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.2355 - accuracy: 0.9514 - val_loss: 0.1958 - val_accuracy: 0.9671
    Epoch 14/20
    180/180 [==============================] - 2s 9ms/step - loss: 0.2149 - accuracy: 0.9535 - val_loss: 0.1829 - val_accuracy: 0.9667
    Epoch 15/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.1938 - accuracy: 0.9570 - val_loss: 0.1662 - val_accuracy: 0.9700
    Epoch 16/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.1809 - accuracy: 0.9582 - val_loss: 0.1478 - val_accuracy: 0.9718
    Epoch 17/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.1665 - accuracy: 0.9608 - val_loss: 0.1375 - val_accuracy: 0.9684
    Epoch 18/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.1552 - accuracy: 0.9621 - val_loss: 0.1265 - val_accuracy: 0.9693
    Epoch 19/20
    180/180 [==============================] - 2s 11ms/step - loss: 0.1441 - accuracy: 0.9630 - val_loss: 0.1164 - val_accuracy: 0.9747
    Epoch 20/20
    180/180 [==============================] - 2s 10ms/step - loss: 0.1364 - accuracy: 0.9666 - val_loss: 0.1104 - val_accuracy: 0.9747



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label="training data")
plt.plot(history.history["val_accuracy"], label="validation data")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fcc4b30a6d0>




    
![png](output_21_1.png)
    


### Model 2: Text Only


```python
text_input = tf.keras.Input(
    shape=(1,),
    name = "text",
    dtype = "string"
)
```


```python
# layers for processing the title
text_features = text_vectorize_layer(text_input)

# Add embedding layer, dropout
text_features = layers.Embedding(size_vocabulary, output_dim = 3, name="embedding2")(text_features)
text_features = layers.Dropout(0.4)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.4)(text_features)
text_features = layers.Dense(2, activation='relu')(text_features)

output = layers.Dense(2, name="fake")(text_features) 
```


```python
model2 = tf.keras.Model(
    inputs = text_input,
    outputs = output
)
```


```python
from tensorflow.keras import utils

model2.summary()
utils.plot_model(model2)
```

    Model: "model_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text (InputLayer)           [(None, 1)]               0         
                                                                     
     text_vectorization_3 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding2 (Embedding)      (None, 500, 3)            6000      
                                                                     
     dropout_4 (Dropout)         (None, 500, 3)            0         
                                                                     
     global_average_pooling1d_2   (None, 3)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_5 (Dropout)         (None, 3)                 0         
                                                                     
     dense_2 (Dense)             (None, 2)                 8         
                                                                     
     fake (Dense)                (None, 2)                 6         
                                                                     
    =================================================================
    Total params: 6,014
    Trainable params: 6,014
    Non-trainable params: 0
    _________________________________________________________________





    
![png](output_26_1.png)
    




```python
model2.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history = model2.fit(train, validation_data=val, epochs=20)
```

    Epoch 1/20


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning: Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)


    180/180 [==============================] - 4s 19ms/step - loss: 0.6907 - accuracy: 0.5299 - val_loss: 0.6852 - val_accuracy: 0.5364
    Epoch 2/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.6636 - accuracy: 0.7319 - val_loss: 0.6298 - val_accuracy: 0.8787
    Epoch 3/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.5893 - accuracy: 0.8307 - val_loss: 0.5279 - val_accuracy: 0.9204
    Epoch 4/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.4990 - accuracy: 0.8798 - val_loss: 0.4294 - val_accuracy: 0.9272
    Epoch 5/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.4213 - accuracy: 0.9020 - val_loss: 0.3582 - val_accuracy: 0.9516
    Epoch 6/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.3631 - accuracy: 0.9121 - val_loss: 0.2960 - val_accuracy: 0.9602
    Epoch 7/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.3200 - accuracy: 0.9188 - val_loss: 0.2558 - val_accuracy: 0.9567
    Epoch 8/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.2850 - accuracy: 0.9242 - val_loss: 0.2293 - val_accuracy: 0.9595
    Epoch 9/20
    180/180 [==============================] - 4s 19ms/step - loss: 0.2611 - accuracy: 0.9284 - val_loss: 0.2031 - val_accuracy: 0.9676
    Epoch 10/20
    180/180 [==============================] - 3s 19ms/step - loss: 0.2408 - accuracy: 0.9313 - val_loss: 0.1800 - val_accuracy: 0.9691
    Epoch 11/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.2269 - accuracy: 0.9288 - val_loss: 0.1676 - val_accuracy: 0.9691
    Epoch 12/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.2123 - accuracy: 0.9371 - val_loss: 0.1555 - val_accuracy: 0.9684
    Epoch 13/20
    180/180 [==============================] - 3s 19ms/step - loss: 0.1982 - accuracy: 0.9368 - val_loss: 0.1382 - val_accuracy: 0.9731
    Epoch 14/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.1908 - accuracy: 0.9388 - val_loss: 0.1335 - val_accuracy: 0.9717
    Epoch 15/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.1822 - accuracy: 0.9383 - val_loss: 0.1325 - val_accuracy: 0.9720
    Epoch 16/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.1768 - accuracy: 0.9361 - val_loss: 0.1191 - val_accuracy: 0.9753
    Epoch 17/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.1724 - accuracy: 0.9398 - val_loss: 0.1058 - val_accuracy: 0.9776
    Epoch 18/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.1653 - accuracy: 0.9404 - val_loss: 0.1061 - val_accuracy: 0.9773
    Epoch 19/20
    180/180 [==============================] - 3s 18ms/step - loss: 0.1639 - accuracy: 0.9401 - val_loss: 0.0997 - val_accuracy: 0.9782
    Epoch 20/20
    180/180 [==============================] - 3s 17ms/step - loss: 0.1585 - accuracy: 0.9399 - val_loss: 0.0978 - val_accuracy: 0.9802



```python
plt.plot(history.history["accuracy"], label="training data")
plt.plot(history.history["val_accuracy"], label="validation data")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fcc40ef4910>




    
![png](output_29_1.png)
    


### Model 3: Title and Text


```python
main = layers.concatenate([title_features, text_features], axis = 1)
```


```python
main = layers.Dense(4, activation='relu')(main)
output = layers.Dense(4, name="fake")(main) 
```


```python
model3 = tf.keras.Model(
    inputs = [title_input, text_input],
    outputs = output
)
```


```python
model3.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history = model3.fit(train, 
                    validation_data=val,
                    epochs = 20)
```

    Epoch 1/20
    180/180 [==============================] - 8s 40ms/step - loss: 1.0807 - accuracy: 0.7412 - val_loss: 0.8972 - val_accuracy: 0.8238
    Epoch 2/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.8153 - accuracy: 0.8341 - val_loss: 0.6897 - val_accuracy: 0.9233
    Epoch 3/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.6401 - accuracy: 0.9288 - val_loss: 0.5430 - val_accuracy: 0.9631
    Epoch 4/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.5208 - accuracy: 0.9593 - val_loss: 0.4383 - val_accuracy: 0.9766
    Epoch 5/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.4267 - accuracy: 0.9735 - val_loss: 0.3624 - val_accuracy: 0.9858
    Epoch 6/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.3540 - accuracy: 0.9797 - val_loss: 0.2913 - val_accuracy: 0.9898
    Epoch 7/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.2939 - accuracy: 0.9842 - val_loss: 0.2426 - val_accuracy: 0.9927
    Epoch 8/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.2490 - accuracy: 0.9837 - val_loss: 0.2031 - val_accuracy: 0.9921
    Epoch 9/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.2139 - accuracy: 0.9854 - val_loss: 0.1734 - val_accuracy: 0.9927
    Epoch 10/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.1824 - accuracy: 0.9874 - val_loss: 0.1494 - val_accuracy: 0.9940
    Epoch 11/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.1607 - accuracy: 0.9876 - val_loss: 0.1281 - val_accuracy: 0.9929
    Epoch 12/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.1408 - accuracy: 0.9880 - val_loss: 0.1161 - val_accuracy: 0.9915
    Epoch 13/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.1216 - accuracy: 0.9896 - val_loss: 0.0978 - val_accuracy: 0.9947
    Epoch 14/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.1098 - accuracy: 0.9904 - val_loss: 0.0926 - val_accuracy: 0.9921
    Epoch 15/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.0972 - accuracy: 0.9906 - val_loss: 0.0795 - val_accuracy: 0.9949
    Epoch 16/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.0895 - accuracy: 0.9900 - val_loss: 0.0719 - val_accuracy: 0.9947
    Epoch 17/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.0786 - accuracy: 0.9920 - val_loss: 0.0666 - val_accuracy: 0.9944
    Epoch 18/20
    180/180 [==============================] - 4s 22ms/step - loss: 0.0724 - accuracy: 0.9904 - val_loss: 0.0542 - val_accuracy: 0.9956
    Epoch 19/20
    180/180 [==============================] - 4s 21ms/step - loss: 0.0676 - accuracy: 0.9909 - val_loss: 0.0494 - val_accuracy: 0.9960
    Epoch 20/20
    180/180 [==============================] - 7s 37ms/step - loss: 0.0627 - accuracy: 0.9913 - val_loss: 0.0469 - val_accuracy: 0.9953



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label="training data")
plt.plot(history.history["val_accuracy"], label="validation data")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fcc41869790>




    
![png](output_36_1.png)
    


### Model Evaluation


```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"

test_data = pd.read_csv(test_url)
test_data

test = make_dataset(test_data)
```


```python
test_evaluate = model3.evaluate(test)
```

    225/225 [==============================] - 3s 12ms/step - loss: 0.0541 - accuracy: 0.9930


### Embedding Visualization


```python
weights = model3.get_layer('embedding2').get_weights()[0] # get the weights from the embedding layer
vocab = text_vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```


```python
import plotly.express as px 
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                # size_max = 2,
                 hover_name = "word")

fig.show()

from google.colab import files
fig.write_html("wordfig.html")
files.download("wordfig.html") 
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="eef5afff-a848-44dc-85b0-88b925cc316c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("eef5afff-a848-44dc-85b0-88b925cc316c")) {                    Plotly.newPlot(                        "eef5afff-a848-44dc-85b0-88b925cc316c",                        [{"hovertemplate":"<b>%{hovertext}</b><br><br>x0=%{x}<br>x1=%{y}<br>size=%{marker.size}<extra></extra>","hovertext":["","[UNK]","said","trump","the","us","i","would","president","people","it","one","state","also","new","donald","states","house","government","clinton","he","obama","republican","could","told","united","in","like","white","campaign","we","last","two","time","news","election","party","first","this","a","former","even","year","country","but","years","hillary","many","that","security","political","may","say","media","make","national","made","get","law","court","since","police","republicans","going","american","presidential","percent","and","back","democratic","administration","bill","support","week","russia","know","america","including","senate","think","public","officials","according","vote","way","take","group","office","trumps","north","federal","they","right","called","foreign","statement","million","military","world","department","want","saying","washington","well","see","you","tax","tuesday","congress","still","much","says","part","russian","another","wednesday","there","day","friday","thursday","minister","if","work","women","go","democrats","asked","policy","war","2016","committee","need","monday","city","americans","deal","next","secretary","rights","china","three","black","help","whether","official","general","never","man","case","around","york","show","on","order","leader","senator","candidate","took","members","come","use","good","countries","without","korea","left","really","report","put","meeting","times","power","end","every","intelligence","she","used","fbi","attack","month","money","investigation","trade","top","change","information","justice","reported","groups","leaders","fact","syria","twitter","long","decision","already","days","plan","iran","business","voters","family","story","conservative","international","far","nuclear","speech","now","here","several","months","likely","however","interview","children","as","so","health","among","place","barack","south","director","clear","must","something","press","agency","program","social","believe","fox","recent","got","move","came","islamic","call","least","sanders","chief","major","john","things","school","issue","immigration","home","is","killed","sunday","might","post","found","him","border","trying","control","reporters","act","though","seen","number","billion","matter","supporters","point","earlier","great","actually","went","later","spokesman","nation","them","today","look","economic","thing","executive","system","working","free","become","past","away","real","keep","all","making","yet","set","march","for","little","win","more","give","added","what","at","democrat","attacks","violence","nothing","muslim","four","stop","big","ever","comment","companies","let","member","legal","july","prime","issues","senior","forces","lawmakers","january","across","taking","defense","local","held","head","following","nations","2015","european","eu","not","gun","expected","talks","opposition","action","human","cruz","no","governor","to","known","woman","continue","care","given","enough","company","sanctions","legislation","when","person","better","nominee","released","others","illegal","process","supreme","force","night","high","history","possible","community","important","wall","june","done","job","wrote","lot","men","open","source","majority","pay","financial","team","anyone","refugees","close","course","reports","life","taken","face","un","syrian","evidence","union","private","response","attorney","question","10","ago","ban","20","wants","run","judge","fight","mexico","budget","plans","second","special","1","despite","gop","conference","staff","email","using","anything","accused","air","university","iraq","debate","letter","saturday","find","watch","efforts","early","someone","race","comments","lives","able","november","agreement","less","putin","while","behind","after","instead","crisis","along","best","future","full","students","sure","within","calling","ryan","mr","five","weeks","current","hard","death","civil","role","region","name","announced","his","due","running","comes","israel","lead","getting","live","effort","congressional","visit","sources","texas","event","service","coming","jobs","center","council","economy","coalition","rules","votes","with","sent","elections","global","citizens","candidates","8","facebook","december","line","comey","saudi","allow","problem","authorities","ties","thousands","muslims","some","emails","young","september","october","paul","chairman","representatives","britain","out","2014","nearly","15","middle","position","criminal","hold","talk","street","daily","wanted","led","leave","politics","army","capital","tell","together","ruling","relations","needs","claims","east","turkey","climate","bush","began","florida","weapons","central","services","immediately","failed","officers","peace","means","late","healthcare","obamacare","april","whose","policies","showed","rather","message","everyone","gave","district","2017","cannot","start","list","outside","based","do","questions","rule","tried","parliament","thought","voting","reform","read","lost","different","elected","speaking","words","up","august","release","immigrants","agencies","racist","try","liberal","germany","access","strong","almost","ministry","latest","received","bad","spending","again","idea","county","workers","millions","threat","enforcement","recently","concerns","bring","hope","everything","cut","conservatives","reason","ahead","stand","laws","protect","always","february","meet","involved","charges","morning","decided","3","freedom","george","funding","poll","makes","allowed","allegations","planned","six","her","industry","of","allies","provide","organization","key","fake","sexual","fire","kind","2","entire","oil","nomination","met","hate","happened","talking","officer","needed","shooting","denied","often","energy","rally","parties","century","looking","side","especially","europe","voted","include","seems","movement","small","claim","shot","adding","situation","market","30","fighting","james","room","either","me","agreed","calls","old","chinese","step","12","vice","insurance","personal","host","west","presidency","near","large","although","request","hearing","actions","worked","true","representative","arrested","western","address","spoke","terrorist","bank","potential","leading","hours","2012","term","confirmed","forward","j","serious","polls","hit","shows","clearly","california","data","return","protesters","missile","area","11","terrorism","decades","s","realdonaldtrump","foundation","crime","water","cases","pressure","feel","tweet","5","wife","interest","wrong","result","biggest","probably","building","simply","review","myanmar","families","travel","appeared","adviser","passed","front","documents","claimed","moscow","commission","dollars","british","paid","fired","declined","continued","alleged","record","soon","board","love","toward","nov","college","turned","spent","brought","signed","mean","details","tillerson","4","relationship","truth","included","short","main","david","proposed","korean","popular","points","became","obamas","article","started","leadership","previously","clintons","saw","primary","bernie","network","aid","forced","influence","turn","attempt","issued","taxes","pretty","pass","account","michael","food","victory","religious","posted","friends","half","level","25","son","final","father","mark","view","mike","fund","incident","longer","hand","child","guy","armed","mccain","independence","helped","german","whole","protest","fear","created","agenda","debt","repeatedly","seeking","21st","website","town","lawyer","giving","these","raised","our","merkel","independent","heard","total","increase","currently","ted","protests","ask","an","reality","mayor","deputy","criticized","remarks","largest","regional","push","described","conflict","respond","education","2013","constitution","third","remain","militants","violent","san","flynn","robert","programs","build","else","published","sign","areas","firm","hundreds","convention","absolutely","hands","example","employees","ambassador","rubio","arabia","pence","phone","similar","cost","living","single","flag","18","speak","lower","iraqi","inside","al","goes","seven","appears","victims","secret","tv","speaker","refugee","telling","spokeswoman","sessions","how","isis","mass","apparently","criticism","changes","troops","tweeted","politicians","christian","100","by","understand","individuals","crowd","carolina","then","discuss","cia","proposal","quickly","japan","project","base","measures","previous","television","risk","radio","asking","urged","online","medical","warned","research","experts","opinion","northern","stay","safety","businesses","problems","mainstream","24","its","focus","events","senators","voter","student","moore","happen","cause","are","died","photo","completely","joe","seek","provided","form","share","safe","johnson","development","13","southern","funds","exactly","prevent","dangerous","50","rep","drug","respect","kurdish","land","numbers","why","nato","island","certainly","prison","page","presidents","internet","interests","were","poor","transition","served","france","concern","church","presidentelect","attention","rate","threats","fellow","16","moment","create","results","gets","certain","society","concerned","trip","14","measure","hear","ready","brexit","6","coverage","considered","corruption","down","committed","named","powerful","parents","ground","false","terrorists","leaving","who","responsible","book","terms","choice","9","consider","kelly","charged","backed","answer","organizations","governments","eight","schools","knew","expect","investment","cities","approved","operations","residents","critical","just","growing","ensure","charge","chance","french","yes","takes","referendum","series","affairs","target","difficult","king","responded","massive","filed","expressed","worst","diplomatic","holding","husband","fraud","because","parts","terror","rhetoric","cuts","class","critics","behavior","authority","favor","democracy","play","dead","7","believed","panel","mother","courts","records","mcconnell","assault","reporter","reached","knows","threatened","low","car","views","offered","maybe","gas","amendment","standing","agree","about","rohingya","22","suggested","repeal","ordered","guns","serve","believes","2011","screen","offer","huge","cabinet","strategy","newspaper","protection","direct","individual","17","impact","progress","virginia","population","negotiations","counsel","god","complete","refused","crimes","paris","sides","environmental","body","supported","statements","send","radical","activists","finally","my","jan","credit","2018","agents","w","sought","killing","domestic","additional","ability","common","beyond","london","abortion","related","raise","28","weekend","remains","opportunity","joint","exchange","27","period","before","word","supporting","ways","status","defend","rest","red","o","labor","2008","19","operation","buy","avoid","regulations","lose","promised","lack","during","citing","cover","chris","caused","sean","scandal","inc","puerto","effect","towards","iranian","turkish","xi","ohio","corporate","significant","showing","includes","continues","reach","attacked","publicly","appear","replace","chicago","test","capture","directly","daughter","regarding","mexican","referring","michigan","21","income","establishment","sense","multiple","arab","opposed","beijing","willing","seem","gone","per","arms","afghanistan","from","announcement","star","perhaps","followed","fiscal","electoral","upon","particularly","growth","costs","sen","join","arrest","worse","necessary","gay","eastern","lawsuit","kids","quite","higher","dnc","sept","looks","jerusalem","deep","facts","considering","approval","worth","socalled","regime","accept","subject","supporter","lies","summit","declared","becoming","israeli","green","communities","2010","communications","spicer","shut","launched","fair","thinks","block","association","putting","jeff","cuba","cooperation","winning","trial","minority","challenge","macron","trust","other","battle","stage","rival","remember","mostly","migrants","lawyers","jr","dc","vladimir","finance","begin","aimed","oct","canada","price","sea","noted","journalists","friend","decide","ended","responsibility","propaganda","legislative","reporting","rise","rich","briefing","site","mind","kim","revealed","probe","homeland","steve","cyber","language","facing","conspiracy","classified","accusations","was","over","joined","most","investigating","caught","ukraine","tough","libya","infrastructure","helping","follow","alliance","soldiers","largely","discussed","manager","sex","blame","reasons","pyongyang","scheduled","goal","break","required","available","couple","ally","scott","de","cast","acting","bannon","alabama","professor","experience","racism","did","deals","version","seriously","emergency","bureau","separate","be","accounts","ruled","moving","meant","estate","26","hollywood","various","shared","powers","amount","rejected","banks","ran","technology","rightwing","pick","tensions","kill","controversial","amid","23","im","both","approach","decisions","coal","will","hill","billionaire","those","pointed","arizona","yemen","pushed","positions","carry","constitutional","aides","played","murder","focused","supposed","seat","property","damage","moved","meetings","african","heart","transgender","lie","tweets","investigations","alone","29","pm","nbc","guilty","conversation","promise","fully","40","appeals","steps","resolution","benefits","addition","jail","changed","60","hurt","bit","removed","embassy","voice","russians","demand","broke","light","condition","allowing","too","iowa","dropped","prosecutors","felt","literally","game","reduce","judges","career","beginning","angry","stated","familiar","appeal","mission","conditions","throughout","identified","erdogan","carried","argued","solution","mueller","strike","lying","hospital","female","faced","doubt","written","claiming","uk","none","increased","hopes","borders","abc","wikileaks","paying","compared","born","annual","islam","jones","present","explain","biden","under","stopped","cited","values","highly","2009","starting","sentence","playing","lady","zone","thats","figure","treasury","positive","opponents","msnbc","below","activities","targeted","rico","delegates","analysis","please","australia","territory","bloc","partner","particular","hannity","fall","management","bid","require","designed","africa","warning","piece","minutes","asia","sort","religion","picture","document","girl","conduct","resources","requests","receive","planning","decade","works","treatment","partners","markets","bangladesh","victim","racial","possibly","internal","average","or","warren","testimony","romney","bringing","pentagon","unless","stories","restrictions","judicial","hotel","campus","bills","housing","dozens","confirmation","brown","age","institute","featured","illegally","happy","affordable","involvement","save","judiciary","entering","dec","pushing","militant","field","faces","defeat","veterans","server","losing","hell","humanitarian","surprise","province","nine","matters","investors","aide","very","miles","drew","closed","basis","admitted","hacking","disaster","arrived","reportedly","possibility","ceo","secure","greater","discussion","sometimes","sales","whatever","reforms","conducted","benefit","audience","taiwan","progressive","ones","loss","islamist","totally","epa","campaigns","allegedly","airport","offensive","leftist","clean","ben","ongoing","kept","ad","suspected","kremlin","behalf","specific","prior","note","worried","t","abuse","sarah","assad","numerous","jim","happening","assistance","thank","spain","prepared","politically","opened","highest","frontrunner","brussels","watching","eric","associated","alternative","stance","intended","study","standards","prosecutor","listen","31","scene","door","systems","nobody","marriage","backing","inauguration","immediate","explained","names","levels","boost","assembly","training","eventually","relief","park","fuel","extremely","citizen","streets","evening","bomb","martin","check","advance","innocent","don","attempted","asylum","seemed","mattis","fighters","built","broadcast","blamed","civilians","vietnam","treated","rates","potentially","parenthood","truly","republic","otherwise","platform","jersey","detained","vowed","seats","search","palestinian","negative","happens","wait","sheriff","paper","carson","pennsylvania","ethnic","dr","correct","cash","attend","thanks","serving","road","neither","convicted","failure","collusion","administrations","ultimately","spend","remove","increasingly","getty","communist","thinking","signs","prove","does","agent","activist","shown","lebanon","keeping","camp","retired","gives","dismissed","tom","raising","oh","lines","confidence","boy","richard","polling","lee","floor","seeing","pakistan","missiles","mention","flint","denies","culture","actual","schumer","production","destroy","looked","involving","failing","drive","detroit","vehicle","minimum","investigators","estimated","drop","donors","date","conway","choose","capitol","aware","session","rape","payments","orders","easy","wounded","projects","prices","can","wonder","liberals","interior","environment","21wire","wisconsin","tells","stood","sitting","injured","girls","diplomats","appointed","unlikely","sell","35","wealthy","their","surveillance","strikes","linked","ballot","off","manafort","limited","figures","christmas","strongly","blocked","yesterday","fine","activity","posts","path","opening","natural","launch","hurricane","contact","analysts","code","players","faith","club","carrying","attempts","overseas","attacking","chair","background","waiting","spread","promote","have","drugs","resignation","limit","appearance","venezuela","sending","narrative","institutions","grand","widely","sweden","successful","simple","film","attended","veteran","tehran","presence","dialogue","proposals","investigate","range","pledged","oregon","difference","andrew","2017realdonaldtrump","navy","congressman","christie","screenshot","places","kushner","heads","grant","obvious","links","knowledge","discrimination","chancellor","businessman","wearing","unclear","products","abe","standard","shift","providing","michelle","draft","construction","youtube","suspect","station","guard","committees","taxpayers","space","reaction","oversight","learned","homes","creating","zero","super","nice","hour","questioned","improve","himself","obviously","nature","mistake","resign","praised","oppose","mitch","banned","returned","practice","visa","temporary","medicaid"],"legendgroup":"","marker":{"color":"#636efa","size":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],"sizemode":"area","sizeref":0.005,"symbol":"circle"},"mode":"markers","name":"","showlegend":false,"x":[-0.002802085829898715,-0.27766549587249756,4.6635284423828125,-1.1373320817947388,2.500884532928467,1.1609642505645752,0.49224498867988586,0.12504522502422333,0.5797924399375916,0.6610637903213501,-0.9529007077217102,-0.708297848701477,-0.24860873818397522,-0.5374919772148132,0.46275588870048523,1.281847596168518,0.5121169686317444,0.7600774168968201,1.8127567768096924,-0.5525918006896973,-0.2626473903656006,-3.0938374996185303,2.093801975250244,0.5917497873306274,2.121166467666626,0.6476337313652039,2.0879039764404297,-2.0016937255859375,0.0538516566157341,0.6614708304405212,-0.3570398986339569,1.4923617839813232,0.933618426322937,-1.2883018255233765,-1.6604245901107788,1.3093595504760742,0.03195976838469505,0.804722785949707,-2.7522695064544678,2.6983423233032227,1.7299892902374268,-3.0793278217315674,1.7740206718444824,0.7038750052452087,2.1978061199188232,1.1758062839508057,-2.421800374984741,0.45292040705680847,-1.6471689939498901,-0.04603278264403343,0.31050559878349304,-0.5438758730888367,0.3918914496898651,-0.08298124372959137,-0.5749468207359314,0.4050627052783966,-0.7849525213241577,-1.156444787979126,0.1332484632730484,0.5123770236968994,0.9577160477638245,0.5480768084526062,-0.8286277055740356,-1.0462108850479126,-3.1654741764068604,2.5807440280914307,1.4359101057052612,-2.2290382385253906,-1.2430049180984497,1.4938225746154785,-1.1842997074127197,-0.11614485085010529,0.48203325271606445,1.3063265085220337,-0.6586824655532837,-1.6644628047943115,-3.582902193069458,1.5235110521316528,1.2066254615783691,-0.19452963769435883,-0.07631545513868332,0.2346828579902649,-0.8542739152908325,-0.15720070898532867,-1.3235423564910889,-0.569419264793396,0.4724700450897217,0.310912549495697,7.385409832000732,-0.1759735494852066,-0.5568282008171082,-1.3879936933517456,-1.4570261240005493,0.9954067468643188,0.8128695487976074,2.6801931858062744,0.5705359578132629,-0.12850981950759888,-1.071326732635498,-1.5891351699829102,-0.6686488389968872,2.259510040283203,-1.7294501066207886,-0.850061297416687,-0.7631315588951111,-2.4833078384399414,1.4123809337615967,3.8944127559661865,0.2499828189611435,0.6033267378807068,-1.2389161586761475,-2.473109245300293,0.35410577058792114,-0.7479053735733032,-1.2277711629867554,4.456071376800537,-0.20839712023735046,-0.9595828056335449,3.5458405017852783,3.8637595176696777,2.778059959411621,-1.0905930995941162,0.29388532042503357,0.6879922747612,-1.8016349077224731,-0.21928837895393372,0.9055464267730713,-0.5671570301055908,-0.009113817475736141,-0.7981846332550049,0.6279759407043457,-0.12095212191343307,3.4879448413848877,0.4454256594181061,-2.01253342628479,0.18690891563892365,1.6522403955459595,0.374761164188385,0.7047156691551208,1.5973442792892456,0.7706318497657776,-1.0031737089157104,0.12712453305721283,1.8623590469360352,0.5567923188209534,0.73642498254776,-0.8261675834655762,-0.9428264498710632,-0.38425400853157043,0.8508976697921753,0.16871437430381775,-0.5998934507369995,0.33284643292427063,-0.7486423254013062,1.2169486284255981,0.9340649247169495,0.5032438635826111,-0.6917108297348022,-0.36838483810424805,-1.0227992534637451,-0.36189359426498413,-0.8051955103874207,0.2633836567401886,-0.10526210814714432,0.03902321308851242,-1.5983967781066895,-2.261542558670044,-1.4048945903778076,-1.0677427053451538,0.18374399840831757,-1.1362018585205078,0.4659167528152466,0.09767188876867294,-1.6570080518722534,-0.48410898447036743,-0.5338358879089355,-0.15531082451343536,-2.0866127014160156,-0.42226773500442505,2.59283185005188,-1.806293249130249,-0.24294379353523254,2.462782859802246,0.8691197633743286,-0.15635308623313904,-0.8443318009376526,0.40002110600471497,0.1002192571759224,1.4071015119552612,0.8259448409080505,-2.904421091079712,-0.971756637096405,2.3794760704040527,0.41140973567962646,-0.28616243600845337,-1.060959815979004,0.642711341381073,0.16932524740695953,-0.4041750729084015,0.223623588681221,0.37234896421432495,-0.1416444331407547,-1.7746187448501587,0.792308509349823,0.7501386404037476,-0.4246281683444977,-0.4581719636917114,-0.05334238335490227,-1.6061979532241821,-2.675412893295288,0.5101754665374756,0.7403173446655273,-0.334282785654068,-1.413464069366455,0.6014100313186646,0.3501307964324951,-0.8967890739440918,-1.7811000347137451,0.44306492805480957,1.3231300115585327,-0.464356929063797,1.8652735948562622,1.4153437614440918,0.3777841627597809,-0.2024136483669281,-0.01524355635046959,-2.0477726459503174,-2.6812756061553955,1.4387656450271606,-0.1364404410123825,0.7212877869606018,-1.6573361158370972,-3.2056262493133545,0.300848126411438,-2.393136978149414,0.020175354555249214,-0.18839477002620697,0.8776260614395142,-0.5075743794441223,-0.16845139861106873,-0.49930089712142944,1.9160895347595215,0.45635107159614563,-1.9042129516601562,-1.4203449487686157,-0.009700293652713299,0.6480634808540344,-0.5498789548873901,0.36663004755973816,-3.1186602115631104,2.9314897060394287,2.0209882259368896,-0.9467054605484009,-1.4226343631744385,-0.11659308522939682,-0.45817238092422485,0.4187685549259186,-0.32412290573120117,0.8305925726890564,3.8933231830596924,-1.0903064012527466,-0.47470009326934814,1.1854380369186401,-1.453829288482666,1.137672781944275,-1.9711090326309204,0.4713124632835388,-1.1785149574279785,1.9818098545074463,-0.683736264705658,-3.0895915031433105,-2.2785539627075195,1.032715082168579,3.1172573566436768,-2.576564073562622,0.07060543447732925,-2.72440767288208,-1.1050490140914917,0.8452379703521729,-1.9721543788909912,-0.036151222884655,-0.24924375116825104,-0.31980380415916443,-0.7605525255203247,-1.1564899682998657,-0.628537654876709,-0.3194881081581116,-0.5321358442306519,-1.2052063941955566,-1.466779351234436,-0.14714375138282776,-0.913008451461792,0.8065134882926941,0.1591910868883133,-0.26903271675109863,-0.1260773092508316,-0.20117595791816711,-4.606427192687988,-0.21907749772071838,1.7562040090560913,-2.657691717147827,0.27255669236183167,0.18834149837493896,1.1047817468643188,0.6910656690597534,-2.3793087005615234,-1.369505763053894,1.0582302808761597,-1.3743767738342285,-0.3282153606414795,-1.2268586158752441,3.711688995361328,2.352219343185425,-2.7440567016601562,-1.5689319372177124,0.09986303001642227,-0.013562869280576706,1.858836054801941,0.0657065287232399,0.44662460684776306,1.508847713470459,1.5114027261734009,-0.7859175801277161,-0.25588223338127136,-0.44199320673942566,-0.12586458027362823,0.7802122235298157,0.8664573431015015,1.6304038763046265,0.854074239730835,0.7597267627716064,0.36596593260765076,1.8776602745056152,1.446439266204834,-2.3608345985412598,-1.1417790651321411,1.7611771821975708,2.4052743911743164,2.7241973876953125,-1.1729035377502441,0.21818827092647552,0.3575226068496704,-0.3611859083175659,0.3223360776901245,-1.4055947065353394,0.11277872323989868,-1.1159204244613647,-1.029022216796875,-1.2757675647735596,-0.4053635001182556,-0.8075212240219116,0.6027980446815491,0.37848514318466187,0.9933261275291443,-0.9450587630271912,-1.5317491292953491,-0.7364935278892517,0.30809730291366577,-0.8712272047996521,1.2041800022125244,-0.8644903898239136,0.18854989111423492,-0.8971033692359924,-0.528732180595398,-1.6468205451965332,0.29517197608947754,-1.1915844678878784,0.9750105142593384,-0.9035524129867554,0.029675941914319992,-0.43234148621559143,-0.5879530310630798,-0.7153015732765198,-0.12561358511447906,0.7729912996292114,-0.33146077394485474,-0.26436716318130493,-0.4674423933029175,1.0867218971252441,-0.4697125256061554,-0.26211026310920715,0.7063565254211426,-0.6413133144378662,-2.955653429031372,-1.3869293928146362,1.2232515811920166,-2.1980478763580322,-1.9216872453689575,-0.22870853543281555,-0.6776809692382812,0.3113062381744385,0.2699342370033264,-0.4228937327861786,-0.9427905678749084,1.0074740648269653,0.7761123776435852,-1.2228825092315674,-0.32902851700782776,-1.3743809461593628,0.2456037700176239,0.29246509075164795,0.2401115596294403,0.6499439477920532,-0.4842100441455841,0.03406990319490433,-0.6516781449317932,0.08221327513456345,0.8825283646583557,0.9210243821144104,0.5427265167236328,1.1119251251220703,-0.892512321472168,0.14008240401744843,0.3146545886993408,-5.010343074798584,1.3622015714645386,0.9043200016021729,-0.3160143792629242,-0.7078404426574707,-1.5860199928283691,2.0149049758911133,0.4619638919830322,0.8574968576431274,0.5149962902069092,0.814296305179596,-0.15014439821243286,1.9838545322418213,-0.5892329812049866,-3.606344223022461,0.12796592712402344,1.16262948513031,-1.4905282258987427,1.3169920444488525,0.6297193765640259,-1.8781075477600098,-0.30243992805480957,-0.246245339512825,0.719083845615387,0.4669506251811981,-0.08935640752315521,0.30427438020706177,-0.5958152413368225,-0.3215678632259369,-0.9169100522994995,1.6134072542190552,0.04803911969065666,-1.2072609663009644,1.099155068397522,-1.3105579614639282,-0.3998102843761444,-2.1954798698425293,-0.9506321549415588,-0.4835614860057831,-0.5697318911552429,-3.6288251876831055,0.42099902033805847,0.5920517444610596,-0.37599503993988037,0.388643354177475,-0.394594669342041,0.7594953179359436,0.8931396007537842,2.7392454147338867,0.15045617520809174,-1.6160770654678345,1.0773578882217407,1.1081628799438477,-0.7900878190994263,-2.8753855228424072,-0.11827994883060455,0.6976687908172607,-1.0226771831512451,-1.673312783241272,-1.3836313486099243,0.9214324951171875,1.1550345420837402,0.28147274255752563,-1.1995506286621094,0.43590468168258667,0.27555227279663086,-1.2237634658813477,0.2669591009616852,0.3424569368362427,-0.7406884431838989,0.8173211812973022,0.9218475818634033,0.9502919316291809,-0.08314263075590134,-0.10643519461154938,-0.44375309348106384,0.3533399701118469,-0.2944425046443939,-1.33187735080719,0.4500691592693329,2.315145254135132,-1.2104746103286743,-1.5471608638763428,-0.5949169993400574,-0.15760937333106995,1.0323760509490967,-0.2952437698841095,-0.40384480357170105,2.2077431678771973,0.569858193397522,0.9043775200843811,-0.36762091517448425,3.129505157470703,-1.107211709022522,0.02894965559244156,-0.21812182664871216,-0.9621490836143494,-0.48905256390571594,0.7334330081939697,3.3485615253448486,2.1974384784698486,-1.1628469228744507,0.42422011494636536,0.2855706810951233,0.8673752546310425,-0.5606616735458374,-0.3306541442871094,-1.2232893705368042,0.8867471814155579,-0.38415637612342834,-0.6728651523590088,-3.6982038021087646,0.11548498272895813,1.1677829027175903,-0.2440658062696457,-0.041205503046512604,1.2676050662994385,2.5686116218566895,-1.5113797187805176,1.3641057014465332,1.895234227180481,0.7832832336425781,-0.7829615473747253,-1.2580175399780273,0.42844492197036743,2.5454697608947754,-0.3795764744281769,-0.7954471707344055,-1.154543161392212,0.795391321182251,-0.41275840997695923,1.9559246301651,0.6420415043830872,1.1243386268615723,0.11264743655920029,-1.5171030759811401,0.8858319520950317,-0.311220645904541,1.5215033292770386,2.360183000564575,0.3675156533718109,-1.1883344650268555,1.4811919927597046,0.36137038469314575,3.30845046043396,-0.8162484765052795,-0.36258235573768616,-1.8936738967895508,-0.5430097579956055,-0.5586838722229004,-2.209843873977661,0.07406792789697647,0.2728863060474396,-1.1698071956634521,0.7168046236038208,-0.18970279395580292,-1.2417680025100708,0.5934889912605286,1.231497049331665,-0.5801895260810852,2.9694983959198,-0.9840766191482544,-0.8451691269874573,1.343841791152954,-3.09055233001709,0.3003840744495392,-0.18766096234321594,-0.4161302447319031,0.8481855988502502,-1.780442476272583,-0.9780651926994324,-0.16767460107803345,-0.9635198712348938,-0.6019435524940491,0.9270057082176208,-2.084746837615967,-0.21565097570419312,-1.9306851625442505,0.7596314549446106,0.2946052849292755,0.00343498308211565,-0.9348365664482117,3.463228702545166,-0.2224109023809433,-1.3465800285339355,-1.6380876302719116,0.528827965259552,-1.3895282745361328,-1.449389100074768,-2.3595192432403564,0.9576647877693176,-0.6980489492416382,-1.019520878791809,-1.086808681488037,-2.6190061569213867,0.5913989543914795,-0.35690414905548096,0.24837756156921387,-1.9785330295562744,-0.07320202142000198,-0.18338580429553986,-1.8384686708450317,2.46063232421875,-1.269433856010437,0.11957743018865585,-0.2437610775232315,-0.4938502907752991,-1.4335726499557495,1.4704780578613281,-1.5875566005706787,1.0899826288223267,-2.277280330657959,-1.7375863790512085,-0.979401707649231,0.14973056316375732,-0.6142356991767883,0.11815200746059418,1.0114712715148926,-1.1722235679626465,-1.4905294179916382,1.4398218393325806,-0.7489063739776611,0.29985982179641724,0.5967813730239868,1.1636459827423096,-1.9912112951278687,0.9024495482444763,0.4981153905391693,-0.03714945912361145,-0.06289723515510559,-0.5028871297836304,0.26565244793891907,0.18158254027366638,-1.5758718252182007,-0.9796391725540161,-3.62471342086792,1.5370895862579346,1.6259628534317017,0.2087099254131317,-2.1017916202545166,-2.4601619243621826,-0.7628601789474487,-1.0431504249572754,0.8745408058166504,-1.4347277879714966,2.0303292274475098,2.3096656799316406,0.3067129850387573,0.7289264798164368,1.6237705945968628,-3.9101290702819824,-0.11442210525274277,-0.7289405465126038,-0.7391675710678101,-0.4420282244682312,-0.8404427170753479,-0.31445422768592834,-3.4436569213867188,0.311098575592041,0.9003196954727173,-2.0832767486572266,-1.244498610496521,2.5872344970703125,0.06776071339845657,1.8547340631484985,0.24187268316745758,1.0006331205368042,-0.2324112504720688,-1.2899880409240723,-0.1780119091272354,-0.12470227479934692,1.748594045639038,0.11140581220388412,-1.0434759855270386,1.2370977401733398,0.514947772026062,0.20562681555747986,1.1473950147628784,0.40877479314804077,-0.8100456595420837,-1.7992085218429565,0.33522674441337585,0.4547085464000702,0.2728930413722992,0.6612842679023743,1.2506736516952515,0.534975528717041,0.12740476429462433,-1.100658893585205,-0.007934145629405975,-1.9068470001220703,1.2537121772766113,0.10594125092029572,0.6776660084724426,0.6150490641593933,-0.5507875084877014,-1.8116651773452759,1.3699272871017456,1.1524497270584106,0.23489165306091309,0.7777584195137024,0.7223610281944275,0.36514565348625183,0.45926204323768616,-0.11630871891975403,-3.8146305084228516,-1.304908275604248,1.099669337272644,-0.25455477833747864,-1.78318452835083,-1.3335092067718506,0.2695750594139099,0.8767709136009216,1.0705902576446533,-0.18591801822185516,-0.2040882408618927,0.8992185592651367,0.0619637705385685,-0.5088379979133606,0.77077317237854,-3.435701370239258,0.6035110354423523,-1.2139612436294556,0.11722114682197571,-0.31114596128463745,1.5296512842178345,0.8158435225486755,-0.15650737285614014,0.24868552386760712,-0.8111934661865234,0.007147070951759815,-0.17688722908496857,-1.1502705812454224,-0.7048434019088745,0.317821204662323,-1.560336947441101,0.6194549798965454,-1.6514463424682617,1.322622299194336,3.17305064201355,-0.09524591267108917,-0.06935091316699982,0.5129815936088562,0.4560101330280304,0.28059372305870056,-0.7698149681091309,-0.6250045895576477,-2.529303550720215,1.7208006381988525,1.7679513692855835,-1.0137336254119873,1.7144871950149536,-0.11619901657104492,0.3143138289451599,2.3294754028320312,-1.81106698513031,-0.9633645415306091,-0.24589182436466217,-0.1201363205909729,-0.26965460181236267,-0.9149596691131592,1.4565180540084839,5.053289413452148,-0.4645824730396271,-0.05047997832298279,-0.6898682713508606,1.0945724248886108,0.5186429023742676,-0.7394692301750183,0.08357603847980499,0.46341758966445923,-1.1526788473129272,0.5320037603378296,-2.244307041168213,-0.10910680890083313,-0.5445277690887451,2.846625328063965,-0.9858680963516235,1.3069325685501099,-0.083646759390831,-0.1682402789592743,0.08405174314975739,0.6996800303459167,4.774011611938477,0.18234655261039734,-0.1983000636100769,0.26783791184425354,-0.2548419237136841,3.7142879962921143,0.02968892827630043,-0.8690201640129089,-1.8796331882476807,-1.2114121913909912,0.6283857822418213,-0.9026380181312561,0.23255538940429688,-0.5216686725616455,-0.5723492503166199,-0.6697383522987366,0.04216943308711052,-2.5822174549102783,0.6052085757255554,0.28692036867141724,0.06617531925439835,0.524864137172699,0.5206689238548279,0.814271092414856,0.9815441370010376,-1.0507546663284302,0.8792579770088196,-0.15337234735488892,0.47878125309944153,0.3580179512500763,0.29004591703414917,-0.1312909573316574,-1.363635540008545,0.03744049742817879,0.04608534276485443,0.3173772394657135,-0.038626883178949356,-0.7896292209625244,-0.6188616156578064,-0.017562834545969963,-2.132538318634033,0.3986566364765167,-1.0378657579421997,2.4200870990753174,0.6168232560157776,0.7183062434196472,-1.4534653425216675,-0.2751552164554596,-0.21006301045417786,-0.0041884868405759335,-1.1476058959960938,0.022979777306318283,-0.12695707380771637,1.5832809209823608,-4.914167881011963,0.8366562128067017,1.0245051383972168,0.33631283044815063,-0.9368194937705994,-1.3787031173706055,0.9549977779388428,-1.1758815050125122,0.7473401427268982,0.7323914170265198,-0.4650333821773529,-0.4208163321018219,-0.08308511972427368,-1.259657859802246,-0.5454632043838501,0.5796387195587158,-0.7693318724632263,1.7888323068618774,-0.9755690693855286,-1.2175321578979492,1.4183335304260254,2.105821371078491,1.2402597665786743,0.9642844796180725,2.0187501907348633,0.32294484972953796,0.10097876936197281,1.4204881191253662,2.0795202255249023,0.4145820140838623,-0.17673753201961517,0.394201397895813,0.7130451798439026,0.11406892538070679,2.507042169570923,-0.9470342993736267,-0.0688258558511734,-0.5271168351173401,-0.12389108538627625,0.39879950881004333,1.3460478782653809,-1.4874184131622314,-0.1300530582666397,0.006280358415096998,1.747082233428955,0.3312852084636688,0.6297212243080139,0.2991470992565155,-1.0124483108520508,0.29850295186042786,-0.9760909080505371,-0.8794187307357788,0.04999341443181038,0.5356329083442688,1.0231223106384277,0.4871983528137207,-0.4464097321033478,0.7850741147994995,-0.762455940246582,-0.5677969455718994,-0.7119438648223877,-1.9161677360534668,-0.059109531342983246,-0.5440319180488586,1.1822601556777954,0.46731001138687134,-0.9726067185401917,-0.15206319093704224,-1.8574579954147339,0.761343240737915,-2.403333902359009,-0.20633475482463837,-1.4395167827606201,1.613410472869873,0.22851811349391937,-1.059813380241394,-1.7029865980148315,2.525797128677368,-0.026029573753476143,-1.8073185682296753,-3.0810775756835938,-0.47447946667671204,-2.9666616916656494,2.0511224269866943,1.784876823425293,-0.0894559770822525,-0.6336867213249207,-0.41809865832328796,-0.6431609988212585,-0.2956615686416626,-1.528502345085144,-0.8527982831001282,-0.9187323451042175,0.5906423330307007,-0.4482366442680359,-0.5082981586456299,0.6884914636611938,-1.3619636297225952,0.9357541799545288,0.9748626947402954,2.0904734134674072,0.5073426365852356,0.10541313886642456,0.9586565494537354,1.3688633441925049,2.6377124786376953,0.15261688828468323,0.3390757143497467,-1.3296693563461304,3.2329020500183105,1.00556218624115,0.31919825077056885,-0.15503722429275513,-0.36919736862182617,0.6279615163803101,1.359377145767212,2.6245410442352295,-0.3411949574947357,0.625071108341217,0.7271360158920288,0.9993333220481873,-1.7864711284637451,-0.08416197448968887,4.842048168182373,0.3300861120223999,0.33836039900779724,0.5750716924667358,-1.5947104692459106,-0.9063577651977539,-0.6552930474281311,-0.6970992684364319,-0.2666687071323395,-1.7204811573028564,2.232266426086426,-2.200429677963257,-1.4028599262237549,-1.9016081094741821,0.7318063378334045,-1.0536824464797974,0.667708694934845,-0.7642197012901306,0.20865856111049652,-0.5033361911773682,0.7013456225395203,0.55998295545578,0.6783549785614014,0.5232677459716797,-1.8635448217391968,-0.06673335283994675,-0.3203049302101135,0.018057115375995636,-4.3913116455078125,1.3234747648239136,0.030863910913467407,2.34475040435791,0.5694712400436401,-0.8465529084205627,-2.5167315006256104,0.14302225410938263,1.5754787921905518,-0.9744302034378052,0.20487846434116364,-1.529348373413086,2.1217212677001953,0.6429013609886169,-0.44489744305610657,4.886119365692139,0.9490853548049927,1.1543493270874023,0.32669854164123535,1.4053328037261963,1.4501067399978638,0.049109071493148804,1.922012448310852,-0.4962328374385834,0.25969961285591125,-1.2178747653961182,0.8431009650230408,0.004644058179110289,-1.6363264322280884,-0.35832035541534424,-0.3547734320163727,-2.0098941326141357,0.11726946383714676,-0.19780589640140533,0.5536237359046936,-0.12364751845598221,-0.8984776139259338,2.0042974948883057,-1.1748026609420776,0.8930370211601257,2.2751059532165527,-0.6702020168304443,1.122610330581665,0.4231559932231903,1.9025170803070068,-0.020486338064074516,0.39883849024772644,1.0054532289505005,0.6723071932792664,0.5382334589958191,0.4522579312324524,-0.9757962226867676,-2.3698458671569824,0.6911502480506897,-1.5917222499847412,0.037580739706754684,0.11919253319501877,0.13727374374866486,0.11063935607671738,-0.7042650580406189,0.15953059494495392,-0.42979520559310913,-0.22103668749332428,1.1907414197921753,-1.1584672927856445,0.0228281207382679,0.9974019527435303,0.6325971484184265,0.2783997058868408,-1.3349298238754272,-0.6935024857521057,1.7876551151275635,0.6144609451293945,0.46195143461227417,0.8560410141944885,1.2888195514678955,-0.09811384230852127,-2.2614779472351074,0.3264651894569397,0.5035807490348816,0.5128504037857056,0.07545273751020432,1.8207833766937256,-1.8366138935089111,-0.8519917130470276,1.8867539167404175,0.08523643016815186,0.07294590771198273,-0.24630895256996155,0.9145940542221069,0.10065440833568573,-1.1832364797592163,-2.113949775695801,-0.3771541714668274,2.4369051456451416,0.0028617260977625847,0.559421181678772,0.49918806552886963,0.5257465243339539,-0.5804104208946228,-1.3259787559509277,1.591046690940857,-2.824411392211914,-0.614979088306427,-0.08722998946905136,-1.1598293781280518,2.2053065299987793,-0.7151236534118652,-0.027605624869465828,-0.37471771240234375,0.6079114675521851,0.13225671648979187,-0.2102697789669037,-0.29268261790275574,0.1402435153722763,1.1264222860336304,0.5666401386260986,0.711466908454895,-1.6828163862228394,0.08830789476633072,-1.0088238716125488,-1.023560881614685,0.05918341502547264,-1.2375032901763916,-0.6913539171218872,-0.6425933837890625,1.6872469186782837,0.6494571566581726,0.15782663226127625,-1.142975091934204,1.0700178146362305,-1.0101145505905151,-0.21913395822048187,-0.5359312295913696,0.296284556388855,2.8642568588256836,-0.5576829314231873,0.09369543194770813,1.2920235395431519,1.194530725479126,-0.5321410298347473,-0.26438844203948975,-0.60247403383255,0.3505103588104248,-2.8235538005828857,0.1631958782672882,-1.9157819747924805,1.470391035079956,0.8293712139129639,2.5709965229034424,0.21116258203983307,-0.11120323836803436,-0.704557478427887,-0.751359760761261,0.47879326343536377,1.7841858863830566,-0.06138301268219948,0.15288661420345306,1.6103864908218384,-0.18217122554779053,-0.0832904726266861,-0.44565317034721375,-1.9332332611083984,0.8695791959762573,-0.1405278593301773,1.3016644716262817,0.5892274379730225,1.1828197240829468,0.413407564163208,1.927815556526184,-0.2590872049331665,-2.4976649284362793,0.7672479152679443,-2.0679328441619873,0.42422083020210266,1.9675920009613037,0.08727338910102844,2.107097864151001,-1.2940418720245361,-0.5836489796638489,2.4743173122406006,0.5138596892356873,0.2123887687921524,1.010643482208252,-0.3336321711540222,0.3316258490085602,-0.349962443113327,1.719084620475769,-0.680175244808197,0.06755760312080383,-0.3048499524593353,-0.3444221615791321,0.06828376650810242,0.8497377634048462,-0.8352015018463135,1.5711296796798706,0.11990825831890106,-0.36408326029777527,1.0387879610061646,0.3972158432006836,-0.9928714036941528,-1.1171047687530518,0.4476372003555298,0.6083100438117981,-0.9857679009437561,-2.0043487548828125,-0.24211379885673523,-2.199050188064575,0.5943753719329834,0.8840947151184082,-0.0958208292722702,-0.19500842690467834,0.6662102341651917,0.9459163546562195,0.5649505853652954,-0.197030171751976,-0.5752371549606323,0.25739142298698425,-0.23826166987419128,2.1100504398345947,-1.1974437236785889,-0.8744779825210571,-0.2316351681947708,-1.0734076499938965,0.23437274992465973,2.2250983715057373,0.8642555475234985,-0.8514577150344849,-0.8670607805252075,-0.8657628297805786,2.1869256496429443,1.992002010345459,-0.08164744824171066,0.20343241095542908,0.3902236819267273,-0.6609790921211243,-0.33959460258483887,-2.036752223968506,0.9609408974647522,-0.28629282116889954,-0.8168963193893433,-0.05375370755791664,0.6817742586135864,-0.8224934339523315,0.14697019755840302,-2.0830130577087402,-0.44566959142684937,-0.1211385503411293,-2.3541998863220215,0.5699999332427979,2.6420114040374756,-1.6853760480880737,-0.6072106957435608,0.7705626487731934,-0.9177855253219604,-0.050766609609127045,-1.9926762580871582,0.9096930623054504,0.3252710998058319,1.9909031391143799,-1.2084532976150513,-2.1017115116119385,-1.3149017095565796,-0.8420053124427795,-0.6744372844696045,1.2615139484405518,-1.262156367301941,-0.3374013304710388,-0.0704503282904625,-1.6655853986740112,0.19232672452926636,1.2553329467773438,0.43641233444213867,-1.7074663639068604,0.3201044201850891,0.6723012924194336,0.14515087008476257,-3.728341817855835,-0.33976128697395325,0.36070969700813293,-1.1466091871261597,0.39015793800354004,0.7257694602012634,1.8693761825561523,0.47324296832084656,-0.9590306878089905,-1.4098042249679565,1.1602786779403687,-2.0046117305755615,2.5666089057922363,-0.9259468913078308,-0.23178033530712128,-0.0925578698515892,-0.3680594861507416,-0.7615331411361694,0.11477857083082199,-0.2135096937417984,0.13202273845672607,-1.7644588947296143,0.2238752543926239,0.4930613040924072,-0.5349588394165039,-1.3720505237579346,0.9329363703727722,0.18890899419784546,-0.8912467956542969,-0.06335774064064026,-0.3156322240829468,0.34674927592277527,0.28132522106170654,0.7135297656059265,-0.4707215130329132,-0.3660341799259186,0.897614598274231,-0.42140766978263855,-1.6842344999313354,-0.05933808535337448,1.4723761081695557,-1.0467008352279663,-1.2968676090240479,1.4727903604507446,1.7640092372894287,0.020562609657645226,1.8248647451400757,-0.0377594418823719,1.2328084707260132,2.5393829345703125,1.0823935270309448,0.6906130313873291,0.43747520446777344,0.9845191240310669,3.797578811645508,-1.3546764850616455,0.5213639736175537,-0.537445604801178,1.6803956031799316,-1.1496108770370483,-2.736382246017456,0.6677756309509277,1.6558195352554321,-0.10839591175317764,1.4276684522628784,2.421421766281128,0.8006498217582703,0.4120016396045685,1.4847183227539062,-0.9771110415458679,0.9321300387382507,-0.9870166182518005,-0.3582228422164917,0.8668517470359802,1.7210735082626343,-0.9657517671585083,1.374084711074829,0.77931147813797,-0.2936970293521881,-1.0715500116348267,2.2317912578582764,-0.0027504104655236006,-1.3890891075134277,0.011395643465220928,-2.3264102935791016,1.5952187776565552,-0.8067401051521301,-0.9079883098602295,0.4130485951900482,0.0843389630317688,-0.328565776348114,-1.3006094694137573,-0.16470927000045776,1.6044667959213257,-1.2651344537734985,-0.2509637475013733,-0.09952366352081299,0.0553048811852932,0.9624869227409363,-1.4248381853103638,1.6453241109848022,0.9900602698326111,0.4834481179714203,0.3569607436656952,-0.8173593878746033,-0.449604868888855,1.819686770439148,1.3392177820205688,1.3739722967147827,0.9286771416664124,0.49469199776649475,0.1973058134317398,-1.4091345071792603,-0.8760910630226135,0.28626519441604614,0.03158436715602875,0.23217017948627472,0.14683902263641357,-0.7349209785461426,0.6345775723457336,0.3969825506210327,1.4077776670455933,-0.3100815415382385,1.02267324924469,0.5795478820800781,-0.3939775824546814,-0.22232195734977722,-0.2767043709754944,-0.04940252751111984,0.49981942772865295,-1.2812777757644653,-1.1695225238800049,0.8928945064544678,0.8571212887763977,-1.1434837579727173,0.6961180567741394,0.9140770435333252,1.3610962629318237,-0.33283066749572754,2.7405195236206055,0.5474271774291992,0.2501145601272583,-0.596197783946991,1.2962474822998047,-0.539337694644928,-1.5506149530410767,-0.8608494400978088,-0.5012919306755066,-0.06738261133432388,-0.9777223467826843,1.2235056161880493,1.8780962228775024,-0.1383754312992096,0.678883969783783,-1.025342345237732,0.9348465800285339,1.350784420967102,-1.1975529193878174,-2.2218024730682373,1.7533869743347168,-0.2560747563838959,5.1146345138549805,1.0182585716247559,0.3504759967327118,0.12532322108745575,0.3425513207912445,-0.5114316344261169,-2.021249532699585,1.1001704931259155,0.5241698622703552,-0.9499829411506653,-1.3092074394226074,0.670183002948761,-0.10383795201778412,-0.07654473930597305,-0.3551003932952881,-0.23436911404132843,1.1071739196777344,0.40463733673095703,0.28943270444869995,0.35899150371551514,-1.9307016134262085,0.11706743389368057,-0.5886719822883606,1.0335478782653809,0.9811708331108093,1.795163869857788,1.9543704986572266,0.7455702424049377,0.5816389918327332,-1.5813425779342651,-0.4592140316963196,0.6260860562324524,-0.7841776609420776,-0.5947715044021606,-1.310455322265625,-1.285927176475525,0.7603197693824768,-0.7015713453292847,0.015294821932911873,1.2470362186431885,0.7725067138671875,0.43711167573928833,0.6439756155014038,0.018491802737116814,-1.0184688568115234,-1.3850915431976318,0.11370815336704254,-0.027494998648762703,0.23062501847743988,0.45260751247406006,-0.2370835244655609,-1.3990423679351807,-0.8837108612060547,-0.4040047526359558,-1.6845322847366333,-0.5205293893814087,-0.17280185222625732,-0.6463016867637634,-0.2767148017883301,-0.5523126125335693,-0.8337709307670593,1.1413896083831787,-0.11552096903324127,2.2518136501312256,0.22652381658554077,-1.9996000528335571,0.24831172823905945,1.2457815408706665,1.2470303773880005,0.31998080015182495,-1.04273521900177,-1.5943922996520996,-2.59480619430542,0.5953322052955627,0.6297631859779358,-0.12594206631183624,1.0598186254501343,-1.6144418716430664,0.29878827929496765,1.7173423767089844,1.4188352823257446,-0.027973953634500504,0.9298986792564392,-0.6456127166748047,-0.614888072013855,-1.5753239393234253,1.3489418029785156,-0.880013644695282,1.4606616497039795,-0.09282613545656204,-0.11165951937437057,-1.9112334251403809,-1.7883602380752563,-1.7769055366516113,-0.5023730993270874,0.9631638526916504,-1.0071110725402832,-1.0404918193817139,-1.1556406021118164,-1.3266183137893677,0.6500048637390137,-0.14920589327812195,0.9720097184181213,0.7289119958877563,-0.5478509664535522,-1.56577467918396,-1.749922752380371,-0.5383217930793762,1.6737644672393799,0.40075886249542236,0.6936988830566406,0.47916701436042786,-0.8495275378227234,0.2945435643196106,0.2947452664375305,1.189241886138916,-0.22101815044879913,-0.7006665468215942,-0.1973598599433899,5.113734245300293,0.009892747737467289,1.2782634496688843,0.7487469911575317,2.2047786712646484,-0.9164828062057495,-2.5139687061309814,-0.3276262581348419,0.14647360146045685,0.6930396556854248,-0.35998547077178955,-1.548676609992981,-2.6485514640808105,1.0598249435424805,0.7967612743377686,1.8405128717422485,0.3554416596889496,-1.6913392543792725,-1.4539902210235596,-0.36840951442718506,1.0338667631149292,1.598598599433899,0.0773109570145607,-0.8598631024360657,2.4684808254241943,-0.8660953044891357,-1.0508602857589722,-0.4852602481842041,2.0942904949188232,-1.1128346920013428,-0.5783314108848572,-1.4615709781646729,-0.08450907468795776,-1.2361791133880615,0.11390330642461777,0.02623407170176506,0.5451177358627319,-0.27128350734710693,-0.36985623836517334,1.7666876316070557,0.8927679657936096,0.7947272658348083,0.9641509652137756,1.4168394804000854,2.371551752090454,-1.1157299280166626,-0.662236213684082,-1.2415043115615845,-0.38184890151023865,-0.29871872067451477,-1.9779188632965088,-0.8752785921096802,-0.36847686767578125,-0.5067744255065918,-0.2647366225719452,-1.0364561080932617,-0.8991903066635132,-0.4606475532054901,1.4921932220458984,-0.7762322425842285,0.15775130689144135,-1.326695442199707,0.9106804132461548,-0.4557226896286011,1.4918413162231445,1.914293646812439,-1.1334819793701172,1.0549466609954834,0.017002582550048828,-1.9573848247528076,0.03265668824315071,-0.47596827149391174,-0.10367201268672943,0.9557152390480042,-0.5362357497215271,0.8021644353866577,0.11988764256238937,1.7403998374938965,-0.41473114490509033,2.36550235748291,-0.5785633325576782,1.6017391681671143,0.44312986731529236,-0.7301068902015686,-0.20403002202510834,-0.42626792192459106,-0.5589873194694519,0.9709243178367615,-1.0396045446395874,2.4252219200134277,0.32475972175598145,-0.09022936969995499,1.7079768180847168,1.255506157875061,1.6593002080917358,0.9654309749603271,0.7061131000518799,1.556208848953247,0.8819679021835327,-1.7123723030090332,-0.21894001960754395,-0.08518888801336288,-0.3745554983615875,-3.4640450477600098,-0.21420811116695404,-0.6112329959869385,0.03409816324710846,-0.0497148223221302,0.35232892632484436,1.342774748802185,0.5892048478126526,-0.855072557926178,1.8047006130218506,-0.30249136686325073,-0.8702353239059448,0.7170299291610718,0.9999097585678101,-1.2808986902236938,0.28053027391433716,-0.3741341233253479,1.3960291147232056,-0.5917555093765259,-0.5646321177482605,0.7221994996070862,-2.6540143489837646,0.9646345376968384,-0.614118218421936,-1.4676775932312012,-0.4127271771430969,-0.43877148628234863,-0.721707284450531,-0.3578311800956726,-0.9187234044075012,1.502468228340149,1.545427680015564,-1.0311729907989502,-0.009044180624186993,-0.5761630535125732,-1.5354825258255005,1.3281792402267456,-2.280726432800293,0.20881153643131256,0.3930462598800659,-0.7446439862251282,-1.2246781587600708,0.6147623658180237,-1.382137656211853,-0.417684942483902,-0.9742984175682068,3.11055588722229,0.0270099900662899,-0.31718602776527405,1.1937834024429321,-0.22122687101364136,1.4541510343551636,1.8987246751785278,-0.41334155201911926,0.4038489758968353,-2.176574945449829,0.8562102317810059,1.1925262212753296,-0.07083219289779663,0.34631744027137756,0.8958420157432556,2.0403263568878174,-2.2226405143737793,0.5543283224105835,-0.8513948321342468,-0.42524635791778564,0.27856308221817017,0.2434290051460266,0.7839917540550232,0.7220969200134277,-0.06151529401540756,0.8471646308898926,-2.548762083053589,0.013490831479430199,1.2119076251983643,1.894042730331421,1.0372389554977417,-0.4636448919773102,-0.4121330678462982,-0.330520898103714,-0.7997214198112488,1.7610533237457275,-0.23756234347820282,-1.0356745719909668,1.0194222927093506,0.3727245032787323,1.9889519214630127,-0.1970604956150055,-1.2057170867919922,0.17180977761745453,-1.7201191186904907,-0.9259042739868164,-0.914673388004303,-0.17013368010520935,-0.2606227993965149,0.7951542735099792,1.493031620979309,0.6665102243423462,0.4333117604255676,0.4595288634300232,0.6010383367538452,0.58216792345047,0.3470143973827362,1.1928110122680664,0.19700953364372253,-1.392490029335022,-1.6496983766555786,0.03159908577799797,-0.35732701420783997,-0.14450915157794952,-0.034342218190431595,1.0384067296981812,1.206217646598816,0.7675008773803711,0.6432322263717651,0.843926191329956,0.020108094438910484,-1.1777929067611694,-0.44343897700309753,-2.3792014122009277,-0.5808668732643127,-0.7596212029457092,0.027524419128894806,2.6947312355041504,-1.4542142152786255,-1.2932602167129517,-0.9816532731056213,0.37527427077293396,-1.5467041730880737,0.03076639585196972,0.9462414979934692,-0.7497384548187256,-0.22568227350711823,-0.1343899667263031,0.772784411907196,3.2254526615142822,0.5523712635040283,0.5626072287559509,0.09321316331624985,0.5280514359474182,-3.3398828506469727,-0.020708179101347923,-0.9386809468269348,0.331940621137619,-1.29544997215271,-1.7379170656204224,-1.3447142839431763,-0.3316783905029297,0.35213083028793335,1.8515329360961914,0.2308456301689148,0.15324363112449646,1.2425123453140259,-0.5177563428878784,1.2704620361328125,-0.979357898235321,0.2826085388660431,-1.9438378810882568,-0.056509051471948624,1.1457390785217285,-1.0510187149047852,1.0453449487686157,0.4188900589942932,0.50460284948349,0.31043368577957153,-0.8196493983268738,1.0192394256591797,-0.8694285154342651,-0.902317225933075,-0.7728033661842346,3.03568434715271,0.2792447805404663,-2.199284791946411,-0.7713994979858398,1.1512705087661743,-2.1285064220428467,-0.06997549533843994,0.4822050631046295,0.0588233545422554,0.7692090272903442,-1.4520246982574463,0.17373202741146088,-0.14125612378120422,-0.1522320657968521,0.42464756965637207,0.179132342338562,-0.5030436515808105,1.0280665159225464,-0.5377358794212341,-1.2573540210723877,-1.1780266761779785,-0.44601863622665405,-0.3572726845741272,-0.29832589626312256,0.39521029591560364,0.2032490074634552,-0.6243404150009155,0.5135579109191895,0.9460492730140686,1.0759657621383667,-0.6936264038085938,-2.689143180847168,-1.5152223110198975,1.4713807106018066,0.23817920684814453,-3.511272668838501,-0.6990903615951538,-1.178797721862793,-0.4322431683540344,-0.9907392263412476,0.9975429177284241,-0.006841940805315971,-0.37850722670555115,0.2642379701137543,0.5760298371315002,0.25819116830825806,0.20725466310977936,0.25629040598869324,-0.3100983500480652,-0.1548183113336563,0.42967694997787476,0.9721721410751343,0.39387398958206177,-0.6918303370475769,-0.09947140514850616,0.17790324985980988,0.8149868249893188,-0.2184142917394638,0.10292908549308777,0.47593700885772705,-2.2983810901641846,0.2628423273563385,-0.19962194561958313,1.3257131576538086,1.1167911291122437,1.3645528554916382,0.2547236382961273,-0.3866919279098511,2.081709146499634,-0.2916310131549835,0.7167789340019226,-0.28519338369369507,-0.22882698476314545,-0.5757014751434326,-0.35820847749710083,0.021551599726080894,0.3185678720474243,0.5814799666404724,-1.1566073894500732,-0.30778834223747253,-0.5130416750907898,-0.18271009624004364,0.8044107556343079,0.43133988976478577,-1.1147695779800415,0.8471367955207825,1.4634413719177246,-0.14509591460227966,-0.15336042642593384,2.0533084869384766,0.3162405490875244,-2.082396984100342,0.45807787775993347,0.018436962738633156,1.8593181371688843,-0.2218409776687622,-0.42881760001182556,-1.0625933408737183,-0.09816376864910126,-0.20801617205142975,-0.5236626267433167,0.5987375974655151,-0.2457321733236313,1.6349947452545166,1.7842538356781006,0.04768228530883789,-0.0023757563903927803,0.07657986134290695,-1.4295369386672974,-0.3415675163269043,-0.9986791610717773,-0.3349646329879761,-0.0992051362991333,-1.3653066158294678,0.6060721278190613,-2.636908531188965,-0.982776403427124,-0.32927969098091125,1.0479955673217773,-1.4077632427215576,-2.446481943130493,1.2726211547851562,-1.1677525043487549,-0.44475239515304565,1.008821725845337,2.090418815612793,0.5132221579551697,0.15291322767734528,1.0061585903167725,0.8191512823104858,-0.4313257038593292,0.972914457321167,0.0943046435713768,0.5947659015655518,0.9522324204444885,0.41010382771492004,-1.199150800704956,-0.6940462589263916,0.5959280729293823,0.30729690194129944,2.379340410232544,-1.5980675220489502,-0.028271157294511795,-0.7838774919509888,0.8043761253356934,-1.7947591543197632,1.4685111045837402,-0.4428163170814514,-1.3025177717208862,-0.7938927412033081,0.1288660764694214,0.6909241080284119,0.8456493020057678,1.5059534311294556,-1.210334062576294,-0.8615511059761047,-0.15140709280967712,-1.2529000043869019,1.0975685119628906,1.8206807374954224,0.4367395341396332,0.1385793536901474,1.4397015571594238,1.037709355354309,0.40071436762809753,-1.0185792446136475,1.14223313331604,-0.19231051206588745],"xaxis":"x","y":[-0.004213146865367889,0.01134403981268406,-0.1361173838376999,-0.013452277518808842,-0.01754852384328842,-0.09462208300828934,0.010368790477514267,-0.047255344688892365,-0.05523470416665077,-0.0008260068134404719,0.04414376989006996,-0.019713973626494408,-0.0484265498816967,-0.07921839505434036,-0.16131649911403656,-0.002872317796573043,0.09607508778572083,0.005589330568909645,-0.04600426182150841,-0.023921305313706398,0.06368249654769897,0.029527772217988968,-0.01871064119040966,-0.01024032011628151,-0.022325484082102776,-0.009255773387849331,0.012605241499841213,0.04828675463795662,0.010559294372797012,-0.025929518043994904,0.006944007705897093,-0.03637794777750969,-0.012899770401418209,-0.1319267451763153,0.09007105976343155,0.03705058619379997,-0.002501250011846423,0.03772317245602608,0.1568775326013565,-0.016851646825671196,-0.06390181183815002,0.07548478245735168,0.010839898139238358,-0.017876848578453064,0.10986178368330002,-0.06670905649662018,0.11276157200336456,0.01809803955256939,0.09833791851997375,-0.0023262298200279474,-0.0834682434797287,-0.08369462192058563,-0.04164184629917145,0.017275141552090645,0.08130348473787308,0.0841342881321907,-0.008738485164940357,0.07430312782526016,-0.059397634118795395,-0.025173533707857132,0.03434808552265167,-0.03950297087430954,0.04402906447649002,0.0025758196134120226,-0.030856160447001457,0.02352757751941681,0.019999327138066292,0.019316140562295914,-0.07553346455097198,-0.018325744196772575,0.06314190477132797,-0.03067130781710148,0.07760422676801682,-0.12749692797660828,-0.07604677975177765,-0.033867623656988144,0.021806634962558746,-0.01435139775276184,-0.0908992812037468,-0.03571193665266037,-0.16583971679210663,-0.03849923983216286,0.04436466097831726,0.004696100950241089,0.05416709929704666,-0.0053595989011228085,-0.00853488128632307,0.05541830509901047,0.053266800940036774,-0.04366911202669144,0.03931097313761711,0.08519808948040009,0.15561190247535706,0.09480968862771988,-0.049775801599025726,0.048692721873521805,0.09564628452062607,-0.04516085609793663,0.04722243919968605,0.027505500242114067,-0.031774330884218216,0.06646490097045898,0.03552524745464325,-0.09565699100494385,0.04203791916370392,-0.03245885297656059,0.011573146097362041,-0.12159806489944458,0.0785912498831749,0.11857884377241135,-0.027996215969324112,-0.02685929834842682,-0.05685104429721832,-0.052859608083963394,0.02937677875161171,-0.09197987616062164,-0.054699286818504333,-0.0306902676820755,0.12773221731185913,-0.1318204402923584,-0.11188003420829773,0.06303904950618744,-0.10822952538728714,-0.05127039551734924,0.010070539079606533,0.09570229053497314,0.07288683950901031,0.056795086711645126,-0.024996861815452576,0.03003065660595894,-0.024622071534395218,0.07705259323120117,-0.03986189886927605,0.043326180428266525,-0.013986536301672459,-0.029483849182724953,-0.04206546023488045,-0.04814673960208893,0.00455688638612628,0.09071961790323257,0.12625449895858765,-0.07548745721578598,-0.08777610957622528,-0.05814725533127785,-0.039742257446050644,0.03459341078996658,0.18615439534187317,-0.07056631147861481,0.028486808761954308,-0.07885431498289108,0.01042716484516859,0.03505735471844673,0.0594320222735405,0.044574372470378876,0.018883924931287766,0.12840431928634644,-0.05522254854440689,0.08122433722019196,-0.051953189074993134,-0.08311405777931213,0.10064593702554703,-0.07031876593828201,0.11971622705459595,-0.06775464117527008,-0.06245027855038643,-0.01086815632879734,-0.07043655216693878,0.1298166960477829,0.039666615426540375,-0.04678864777088165,0.02107934094965458,-0.014338863082230091,0.018981540575623512,-0.057545341551303864,0.0185939222574234,0.1277281790971756,-0.00924378726631403,-0.116452157497406,-0.040113676339387894,0.010318621061742306,0.0439780093729496,-0.058167800307273865,-0.12889409065246582,0.0712478905916214,-0.008029334247112274,-0.025560664013028145,-0.0751691684126854,-0.035884469747543335,0.0886416882276535,-0.048506610095500946,0.09542704373598099,0.017872871831059456,-0.0036431942135095596,-0.04470062628388405,0.09671112895011902,0.09698611497879028,-0.12525878846645355,0.011714654974639416,0.06592794507741928,0.04285725578665733,-0.0038684476166963577,0.1280536949634552,-0.17035435140132904,-0.1578575223684311,-0.0848216861486435,-0.05692227557301521,-0.007309814915060997,-0.06134667247533798,-0.014906992204487324,-0.051868632435798645,-0.052189383655786514,0.017355741932988167,0.012949974276125431,-0.048008885234594345,0.06530186533927917,0.10162908583879471,-0.08505192399024963,0.1163453459739685,-0.019536154344677925,-0.042326923459768295,0.07590878009796143,0.11392945796251297,0.1379496157169342,-0.07188219577074051,0.02994196116924286,0.07600279897451401,0.07253562659025192,-0.020252900198101997,0.08197376877069473,-0.11079957336187363,0.012767044827342033,-0.02102411910891533,-0.11700457334518433,-0.06820403784513474,-0.052549224346876144,-0.011426836252212524,-0.004280818626284599,0.07149152457714081,-0.008181043900549412,0.15565967559814453,0.014925585128366947,-0.028423873707652092,0.02523450367152691,-0.09666883945465088,0.14418308436870575,0.03159680590033531,-0.02052830345928669,0.09094616025686264,-0.05068084970116615,0.007488416973501444,0.0903954952955246,0.038227710872888565,-0.0020827471744269133,-0.08424591273069382,-0.009545501321554184,0.07869771122932434,0.008946780115365982,-0.014863886870443821,-0.005345627199858427,-0.06774590164422989,-0.07247806340456009,0.06135788559913635,0.0509674996137619,0.19520971179008484,-0.1079920157790184,-0.034464191645383835,0.06925816833972931,-0.03138205036520958,-0.070412278175354,0.0660238042473793,0.03943905606865883,0.039975184947252274,-0.010836117900907993,-0.05219012871384621,-0.013849547132849693,-0.019838476553559303,-0.05686404928565025,0.08531692624092102,-0.05328119173645973,0.028561750426888466,0.09238400310277939,-0.02337723597884178,0.06127163767814636,0.004482896067202091,0.05351671949028969,0.051266517490148544,0.008250702172517776,-0.01898142881691456,-0.07177034020423889,-0.07868063449859619,-0.14075513184070587,0.08043520152568817,0.05922475457191467,-0.018712973222136497,-0.05853565037250519,-0.04588894546031952,0.10906955599784851,-0.05578019097447395,0.05828307196497917,0.03577802702784538,-0.04199528321623802,0.0943869948387146,-0.027328744530677795,-0.09931043535470963,0.088341623544693,0.015887822955846786,-0.11101426929235458,0.013744299300014973,0.026984544470906258,0.0488010048866272,0.014039658010005951,0.07836943864822388,-0.06419584155082703,-0.042315855622291565,0.10742708295583725,-0.0006890935474075377,0.029895927757024765,-0.13644693791866302,0.009594431146979332,-0.053854718804359436,-0.1330224722623825,-0.10776319354772568,-0.0747624859213829,-0.17713503539562225,-0.1413549780845642,-0.06149587035179138,-0.006597633007913828,-0.030795129016041756,0.028295069932937622,-0.05904484912753105,0.010327626019716263,0.10439176112413406,-0.021841993555426598,-0.13718411326408386,-0.038712359964847565,-0.0331881083548069,-0.0877942219376564,-0.19063128530979156,-0.08230305463075638,-0.029663369059562683,0.18469129502773285,0.03485168516635895,0.07249122858047485,-0.0752355083823204,0.09144534915685654,0.04594581574201584,0.06924878060817719,0.00899172481149435,-0.0262217428535223,-0.10190412402153015,0.023891370743513107,-0.03749481961131096,-0.09181670844554901,-0.03919634595513344,0.18024201691150665,-0.06469222903251648,0.06714818626642227,-0.06126344949007034,0.11101201176643372,0.015946438536047935,-0.07008887082338333,-0.08746015280485153,0.06715785712003708,-0.05829739198088646,-0.021919546648859978,0.1447969526052475,-0.07817364484071732,-0.12127525359392166,-0.08066709339618683,-0.12135569006204605,-0.014678740873932838,-0.01828126050531864,-0.021859701722860336,0.07341999560594559,0.09751985967159271,0.010746655985713005,-0.04844321683049202,-0.08092053234577179,-0.07059679925441742,-0.038021937012672424,-0.08631374686956406,0.12424878031015396,0.08693891018629074,0.16884845495224,-0.010932128876447678,-0.012336376123130322,-0.005988956429064274,0.053172510117292404,-0.027429215610027313,-0.0032849472481757402,0.05483304336667061,-0.050696469843387604,-0.01886424981057644,0.04692604020237923,-0.13214357197284698,0.24219486117362976,-0.012357513420283794,0.06496677547693253,0.06273568421602249,0.061668843030929565,0.09571940451860428,-0.035677578300237656,0.08556373417377472,0.1055319532752037,0.12289519608020782,-0.03680809214711189,-0.02623800002038479,-0.042256783694028854,-0.01052863895893097,-0.03162379190325737,-0.03335346281528473,0.04041368514299393,0.011802196502685547,0.06399314105510712,0.0666809231042862,0.029868070036172867,-0.11229625344276428,-0.04448183625936508,0.05953311175107956,0.2104620486497879,-0.08164048194885254,-0.0359969362616539,0.05329152196645737,-0.10917399823665619,-0.026239655911922455,0.012676895596086979,-0.10954850167036057,-0.12433377653360367,-0.05470679700374603,-0.01769334264099598,0.06951465457677841,-0.09627535194158554,0.03902346268296242,0.056544046849012375,-0.012943015433847904,-0.08466483652591705,-0.04200700670480728,-0.1604202389717102,0.029608268290758133,-0.033572737127542496,-0.04214409738779068,0.1105792224407196,0.0009380450355820358,0.0939403623342514,0.21045230329036713,0.007075270172208548,0.03839998319745064,-0.021449122577905655,-0.06117739528417587,0.03482429310679436,-0.005838769022375345,-0.1525791734457016,0.040091026574373245,-0.16884392499923706,-0.029095903038978577,0.018011851236224174,-0.058447081595659256,0.07968967407941818,0.019090179353952408,0.0664195865392685,-0.05442067235708237,0.003207758069038391,0.058195751160383224,0.11202666163444519,-0.015221305191516876,0.10536246001720428,-0.02541486732661724,0.08255351334810257,-0.04853679612278938,-0.037040237337350845,0.06473620235919952,-0.042676836252212524,-0.0748787447810173,-0.030622586607933044,-0.0819554254412651,0.06935941427946091,-0.019766513258218765,0.06040780991315842,-0.06416863203048706,-0.0892629474401474,-0.10945403575897217,0.0853012204170227,-0.12143073976039886,-0.014752593822777271,0.09258639812469482,-0.09964160621166229,0.005960008595138788,-0.030057240277528763,0.061886511743068695,-0.017891226336359978,0.083001509308815,0.0901995524764061,0.022638773545622826,-0.05577567219734192,0.11269216984510422,-0.06566653400659561,0.09663302451372147,-0.05036628618836403,-0.00039024424040690064,-0.0016448965761810541,0.12555545568466187,-0.0501052550971508,0.00291002937592566,-0.0020500533282756805,0.06498472392559052,-0.051403868943452835,0.11741725355386734,-0.04845964536070824,-0.10212627798318863,0.02908450737595558,-0.07599017024040222,-0.10457457602024078,0.015105458907783031,-0.0409795306622982,0.0005085747689008713,-0.0011743864743039012,-0.03100774995982647,-0.009978597052395344,0.024894066154956818,-0.05044296011328697,-0.07730261236429214,-0.027780860662460327,0.03704913333058357,0.07126179337501526,-0.0079435920342803,-0.030529644340276718,-0.08157754689455032,-0.007312017492949963,-0.10670759528875351,0.057550083845853806,-0.12446505576372147,-0.07612574845552444,-0.17056553065776825,0.005648529157042503,-0.026473598554730415,0.006686771754175425,0.11550696194171906,-0.01530302781611681,-0.09431569278240204,-0.0008877859218046069,-0.03315700218081474,0.05901268869638443,-0.0852951854467392,0.031835418194532394,0.001984117552638054,0.037756823003292084,0.08287004381418228,0.10121030360460281,-0.09758001565933228,-0.0751173123717308,-0.07161609828472137,-0.06493856012821198,0.04692917317152023,0.025829492136836052,0.0778362974524498,-0.004641637206077576,0.07085643708705902,-0.053286828100681305,-0.010429421439766884,0.16820105910301208,0.11324656009674072,-0.045969996601343155,0.1053043007850647,0.006453096866607666,0.15727286040782928,0.023059314116835594,-0.023202938959002495,0.018818655982613564,0.006405050400644541,-0.012154058553278446,-0.05152216926217079,-0.09352154284715652,-0.029567215591669083,0.10231532156467438,-0.02539292722940445,-0.15859739482402802,-0.07330456376075745,0.0903417244553566,-0.11659153550863266,0.07703720778226852,0.029157020151615143,0.03414280712604523,-0.05558136850595474,0.08929167687892914,0.013375610113143921,0.06389839202165604,-0.04097030311822891,-0.039748452603816986,-0.008013889193534851,0.0530441552400589,-0.018134359270334244,-0.00428855000063777,-0.01958523690700531,0.0358424112200737,-0.014443933963775635,-0.015461018308997154,-0.0730152353644371,0.06725823134183884,0.0145593686029315,0.08946528285741806,-0.00817971583455801,0.044366393238306046,0.04160362109541893,0.033566683530807495,0.0857287049293518,0.009452953934669495,0.02194591425359249,0.0178813673555851,-0.01498325727880001,-0.012773023918271065,0.0924420952796936,0.041843101382255554,0.05644601210951805,-0.016353001818060875,0.00821680761873722,0.13986517488956451,-0.08629176765680313,0.06387573480606079,-0.07004762440919876,0.017951125279068947,0.1854739785194397,0.021412890404462814,-0.003368438920006156,0.014831693843007088,0.027033381164073944,0.04011911526322365,0.07032360881567001,0.10990381240844727,-0.10999488085508347,0.0541907474398613,-0.030241455882787704,0.06121911481022835,0.08573969453573227,0.03594646975398064,0.018145686015486717,0.016828440129756927,0.015684230253100395,-0.010593206621706486,-0.12596730887889862,0.06568234413862228,0.06587618589401245,-0.01348629780113697,0.15033771097660065,-0.10370665043592453,0.07166656851768494,-0.06552532315254211,-0.04367292672395706,-0.1171000674366951,0.27166640758514404,-0.025367753580212593,-0.02212386205792427,0.003633268876001239,0.06019551306962967,-0.06941525638103485,-0.10841072350740433,0.0656113401055336,-0.008682440035045147,0.057897139340639114,0.020579172298312187,-0.15540137887001038,0.08622348308563232,-0.05493119731545448,-0.1517079770565033,0.04869360849261284,-0.05200570821762085,-0.06183524429798126,-0.041458096355199814,0.15536589920520782,-0.06793279200792313,0.0977746844291687,0.012673275545239449,-0.020242048427462578,0.03853675350546837,0.050968367606401443,-0.04431025683879852,0.05730229988694191,0.1423361599445343,-0.1412152349948883,-0.02330905757844448,-0.031956590712070465,-0.006907703820616007,-0.038582947105169296,0.012879425659775734,0.003429474076256156,-0.058506663888692856,-0.011194577440619469,0.03616191819310188,-0.03341732174158096,-0.004336861427873373,0.058285973966121674,-0.004036322236061096,0.013866854831576347,-0.009752636775374413,-0.03186315298080444,0.08901448547840118,-0.03604196757078171,0.07024111598730087,0.008896691724658012,0.08369051665067673,0.1331828087568283,-0.02989264205098152,0.015561896376311779,-0.02566436305642128,-0.005537465680390596,-0.004793690051883459,0.09097679704427719,0.0325445793569088,-0.08348057419061661,0.03374780714511871,-0.015373575501143932,0.0016576681518927217,0.042011093348264694,-0.03346887603402138,-0.011574912816286087,0.034589700400829315,0.0635974332690239,0.1676531434059143,0.019085409119725227,0.021209530532360077,0.08674055337905884,0.022874198853969574,-0.04904714599251747,-0.03412957489490509,0.07323023676872253,0.08118294924497604,-0.08006716519594193,0.009802596643567085,0.059957701712846756,-0.1761172115802765,0.0387142039835453,-0.061847228556871414,0.01102753821760416,-0.018486229702830315,-0.014038017019629478,0.080473393201828,-0.06367354094982147,-0.04594512656331062,-0.030048463493585587,0.09939255565404892,0.11060786247253418,0.09640495479106903,0.0745643824338913,0.08336388319730759,-0.004068846348673105,0.11244063824415207,-0.015081485733389854,0.006364957429468632,-0.20414403080940247,-0.16950581967830658,0.038992978632450104,0.058337051421403885,-0.13944999873638153,-0.0724194124341011,0.05817718803882599,0.020882224664092064,0.0050671836361289024,-0.05304235965013504,-0.0915282592177391,0.031638253480196,0.1348465234041214,0.07300502806901932,-0.12188698351383209,0.04531439393758774,0.02994554117321968,0.010442729108035564,-0.09716758131980896,0.048023298382759094,0.04166704788804054,0.049300666898489,0.07249818742275238,0.024773577228188515,-0.04917586222290993,0.04457048326730728,0.08955744653940201,-0.002978881588205695,-0.07455997169017792,0.016907021403312683,0.0668017789721489,-0.015358016826212406,0.01854318007826805,0.009234903380274773,-0.09542085230350494,0.027536295354366302,-0.17683085799217224,-0.07127328962087631,0.10436324775218964,-0.024931926280260086,-0.09892262518405914,0.0006929219234734774,-0.0011848846916109324,-0.020408131182193756,-0.0003251918824389577,-0.05901230499148369,-0.03854476287961006,0.04391012713313103,0.041897252202034,-0.07422186434268951,-0.02087554708123207,-0.01021581795066595,-0.001167918206192553,0.06902296841144562,0.04049651324748993,-0.05312944948673248,0.044240616261959076,-0.0029937506187707186,0.030248388648033142,0.10739541798830032,-0.08631018549203873,0.0531894713640213,0.12295714765787125,-0.022035036236047745,0.08721331506967545,0.05293158441781998,0.14718182384967804,0.0005058148526586592,-0.029341042041778564,0.029336407780647278,-0.07086201757192612,-0.01182783953845501,0.10430332273244858,0.0982128158211708,0.10057633370161057,0.1707950383424759,-0.09813304245471954,-0.05574188381433487,-0.072152279317379,0.09273410588502884,-0.12424088269472122,0.07981648296117783,0.10944048315286636,0.04825286939740181,-0.0037496837321668863,-0.09761356562376022,-0.05290842056274414,-0.04216647893190384,0.11570308357477188,0.0014190897345542908,0.12724776566028595,0.0830920860171318,0.016645027324557304,-0.04681480675935745,-0.018742823973298073,-0.10356949269771576,-0.02379556931555271,-0.10619667172431946,-0.0264121126383543,0.008053574711084366,-0.059653140604496,-0.11886217445135117,-0.00605236180126667,0.10081905126571655,-0.05248773470520973,0.05756485462188721,-0.09188135713338852,-0.09170906245708466,-0.014413110911846161,-0.047272611409425735,0.10793092846870422,0.015491521917283535,-0.01422963384538889,-0.06348659843206406,0.002750667277723551,0.1573885679244995,-0.0003048857906833291,-0.02325831726193428,-0.050824884325265884,-0.0034214043989777565,0.09009337425231934,0.004553719889372587,-0.11837856471538544,0.014617515727877617,0.023678092285990715,-0.08138871192932129,-0.06222258135676384,0.0033999986480921507,-0.11526086926460266,-0.05030599981546402,-0.00030544892069883645,-0.014029808342456818,0.13402637839317322,0.038051631301641464,0.06623801589012146,-0.06403804570436478,0.0004678584518842399,-0.013092990033328533,-0.021788744255900383,-0.0493117980659008,0.013925635255873203,-0.07404842227697372,0.1118568703532219,0.004250133875757456,-0.02398183010518551,0.028274454176425934,-0.08254922181367874,0.06988932937383652,0.029056647792458534,0.0923122689127922,0.06304270029067993,0.09207286685705185,-0.020096376538276672,0.05523228645324707,-0.010344309732317924,0.02804238349199295,0.05132443085312843,-0.11916408687829971,0.05386469513177872,-0.012900011613965034,-0.06823316961526871,0.057226914912462234,-0.02581501007080078,0.021601038053631783,0.04185048118233681,0.06369920074939728,0.01638493500649929,0.00909377634525299,-0.0445062629878521,-0.03307369723916054,-0.021221119910478592,0.11369053274393082,0.05997521057724953,0.0889497697353363,0.06805885583162308,-0.05069112032651901,-0.025198735296726227,0.14708349108695984,-0.10256093740463257,-0.07131746411323547,0.0696013942360878,0.00710152555257082,0.04606287181377411,-0.10168197751045227,-0.025178465992212296,0.0516531802713871,-0.04741006717085838,-0.10166379809379578,-0.13443373143672943,-0.10338850319385529,-0.07697459310293198,0.029737325385212898,-0.08011945337057114,0.11082928627729416,0.009779690764844418,0.04101255536079407,-0.06432005763053894,0.04635806381702423,0.07081405818462372,-0.04288581758737564,0.04337051138281822,0.03404754400253296,-0.11198931187391281,0.13601653277873993,-0.10031658411026001,-0.02671903744339943,0.026481127366423607,0.050700943917036057,0.17692062258720398,0.1559024453163147,-0.08743792027235031,-0.08826038241386414,0.0371866449713707,0.06999867409467697,-0.04408986121416092,-0.05900707468390465,-0.0794922336935997,-0.051418643444776535,-0.008246304467320442,0.009407306089997292,-0.007883721962571144,-0.0672370195388794,0.11458099633455276,0.07292789220809937,0.015234915539622307,0.030254671350121498,0.05058072507381439,-0.09389324486255646,-0.03804410994052887,0.06648477166891098,-0.005853955168277025,-0.05661550164222717,0.0591200515627861,-0.015798497945070267,0.04129258543252945,-0.030292611569166183,-0.03323964774608612,0.07205802947282791,0.011925511062145233,-0.035824380815029144,0.029678406193852425,0.03681422770023346,-0.015553483739495277,-0.05408133938908577,0.08436769247055054,0.13079150021076202,0.10629700869321823,-0.06952148675918579,-0.06178150326013565,0.11390239745378494,0.00019117715419270098,0.0458771176636219,-0.019144050776958466,-0.08415305614471436,-0.006410024128854275,-0.005613967310637236,0.09131865948438644,0.10364317893981934,-0.09031576663255692,0.08371143788099289,0.025486906990408897,0.0991869792342186,0.021453646942973137,-0.01553594134747982,0.0559169240295887,0.03734366223216057,-0.018662618473172188,-0.018779659643769264,-0.07822679728269577,0.06590823829174042,0.008648240007460117,0.06968075037002563,0.055984385311603546,-0.14757144451141357,0.005199712235480547,0.0369110144674778,-0.03123350627720356,0.029913725331425667,0.0012494541006162763,0.010603601112961769,-0.06799323856830597,-0.064137302339077,-0.08041207492351532,-0.0032923587132245302,0.022284062579274178,-0.07611928880214691,0.0014384833630174398,-0.032851532101631165,-0.006518005393445492,-0.061085399240255356,-0.009796860627830029,0.08869478106498718,-0.06745791435241699,0.08824992924928665,0.09970074892044067,0.00302300532348454,0.012923816218972206,-0.06525948643684387,-0.11525110900402069,0.04310353845357895,0.06764331459999084,0.05697562173008919,0.08394303917884827,-0.07006414234638214,-0.022447194904088974,-0.09869858622550964,-0.14479412138462067,-0.06442959606647491,-0.007372187916189432,-0.034434519708156586,-0.0636429712176323,0.013258092105388641,0.10644898563623428,-0.10455858707427979,-0.09460128843784332,0.013186548836529255,-0.07401391863822937,-0.027168255299329758,-0.08817705512046814,-0.01163477823138237,-0.06970121711492538,-0.01925106719136238,0.08313359320163727,0.051767244935035706,-0.003357533598318696,0.0818934217095375,-0.07562205195426941,0.03930738940834999,-0.0322541780769825,0.05987933650612831,-0.027632612735033035,0.04113060235977173,0.03350887447595596,-0.022325709462165833,0.0944218784570694,-0.043848201632499695,0.027865389361977577,-0.03329288214445114,0.05806034803390503,0.11085657775402069,0.020370159298181534,-0.19114242494106293,0.15267495810985565,0.10137336701154709,0.05194597691297531,0.05292757973074913,-0.06293975561857224,-0.11284035444259644,-0.03080318123102188,0.20134063065052032,0.1040087565779686,-0.03725974261760712,-0.1205725446343422,0.024927087128162384,-0.11374583095312119,-0.007453012280166149,0.05995478853583336,-0.012626167386770248,0.09600408375263214,0.04354742914438248,-0.024300504475831985,-0.023131053894758224,-0.05595739185810089,0.00393717223778367,0.0984824150800705,-0.06299696862697601,-0.017688341438770294,-0.06795289367437363,-0.019132371991872787,0.050027914345264435,0.04855220764875412,-0.05649368837475777,-0.031995806843042374,0.031819362193346024,-0.11612793803215027,0.016986403614282608,0.06930316239595413,0.11809824407100677,0.01049171295017004,-0.03443749248981476,0.03901487961411476,0.025440342724323273,0.019490206614136696,-0.0038126364815980196,-0.08295650780200958,-0.059568215161561966,0.022612284868955612,0.05290541797876358,-0.06722036004066467,-0.056083474308252335,0.05910605564713478,-0.14088647067546844,0.09061475098133087,0.0026682105381041765,0.1680896133184433,-0.03053244762122631,-0.00023367023095488548,0.12818799912929535,-0.04523730278015137,-0.02798457257449627,0.03981281444430351,-0.08260095864534378,-0.05946188047528267,0.016559090465307236,-0.05129236727952957,-0.004952060058712959,0.0184148158878088,0.028613751754164696,0.10616109520196915,0.06402502208948135,-0.043766994029283524,-0.09703618288040161,0.05006750673055649,-0.008234018459916115,0.09893609583377838,0.008868839591741562,0.001498495228588581,-0.028275247663259506,0.13698649406433105,-0.0750758945941925,-0.02497662790119648,-0.0864640548825264,0.02332712896168232,-0.06193685159087181,-0.06015583872795105,-0.06501412391662598,0.037874456495046616,0.05753498524427414,-0.09731142222881317,0.07576905936002731,0.13626976311206818,0.1487860381603241,-0.11858272552490234,0.016749992966651917,0.09381145983934402,0.12297935783863068,0.06798854470252991,-0.013602961786091328,0.0027242463547736406,0.027066348120570183,-0.021778056398034096,0.026219148188829422,0.1514699012041092,0.01696467213332653,0.1370810568332672,0.04730084165930748,-0.026123424991965294,0.09340576082468033,-0.012087657116353512,0.007430482655763626,-0.03960248455405235,-0.08365947008132935,0.028808576986193657,0.14488236606121063,0.016933651641011238,-0.02629990316927433,0.001849711057730019,-0.018492983654141426,-0.07129260152578354,-0.00470033660531044,-0.10674306750297546,0.009760738350450993,-0.047133270651102066,0.11220605671405792,0.056969307363033295,-0.05390626937150955,0.018731262534856796,0.058707404881715775,-0.007126195356249809,0.000353806943167001,0.08275288343429565,-0.060659099370241165,0.07672669738531113,-0.0590965636074543,0.053859908133745193,0.02034415304660797,-0.17997342348098755,-0.003580714575946331,-0.05611920356750488,-0.09759896993637085,-0.031374040991067886,0.03397257253527641,0.061182886362075806,-0.07167182862758636,0.012650514021515846,-0.06841716915369034,0.1011570543050766,0.0346335768699646,0.007252898532897234,-0.013521894812583923,-0.07237494736909866,-0.03464467450976372,-0.24259260296821594,0.04846923425793648,-0.03474552556872368,-0.024520503357052803,-0.05062439665198326,-0.0341169610619545,-0.060217928141355515,0.0562446229159832,0.09609383344650269,-0.13945814967155457,-0.08462414890527725,0.05412120372056961,0.0256954412907362,-0.016007225960493088,-0.009200629778206348,0.06675856560468674,-0.015631893649697304,-0.05992889404296875,-0.009759059175848961,-0.08938753604888916,0.08803420513868332,-0.0068547590635716915,-0.16620823740959167,-0.049385469406843185,0.042243920266628265,-0.11315719038248062,-0.11587261408567429,0.020512016490101814,0.017374536022543907,0.1025867611169815,0.024541467428207397,-0.05198327824473381,-0.07161306589841843,-0.06539282202720642,-0.04796129837632179,0.07856600731611252,-0.10302390158176422,-0.12895113229751587,-0.058297596871852875,0.0003701925161294639,-0.013656307011842728,-0.0038418322801589966,-0.013557183556258678,0.09144538640975952,0.007405386306345463,0.015031155198812485,-0.0806288942694664,0.0405331626534462,-0.010810487903654575,-0.039559606462717056,0.03381134942173958,0.03497569262981415,-0.010755246505141258,0.029993558302521706,-0.02959442138671875,-0.15949462354183197,0.08640293031930923,-0.04988863691687584,-0.06837955862283707,-0.1033799797296524,-0.11010633409023285,-0.18277300894260406,0.07213985174894333,0.02866504341363907,-0.006492150481790304,-0.018529312685132027,0.0885540172457695,-0.04993802681565285,-0.025092465803027153,0.03198375552892685,-0.013166113756597042,-0.013709410093724728,0.0998368039727211,-0.09859421104192734,0.06460021436214447,0.012733014300465584,0.10742393881082535,-0.10348975658416748,-0.04363379254937172,0.13902565836906433,0.16273607313632965,0.12031952291727066,0.04048800840973854,0.024451283738017082,0.057117290794849396,-0.09053903818130493,-0.05899728089570999,0.07784464955329895,-0.09335404634475708,-0.00227754982188344,0.048012733459472656,-0.005006325431168079,-0.0009151711128652096,0.00828603282570839,-0.22152283787727356,-0.021846706047654152,0.14208191633224487,-0.09016725420951843,-0.01386721059679985,-0.012753846123814583,0.04341689869761467,0.0007578277727589011,-0.046096086502075195,-0.12194469571113586,0.13045255839824677,0.023639343678951263,0.028741884976625443,-0.02987481839954853,0.03449694812297821,0.0858059898018837,-0.04148764908313751,0.13750813901424408,-0.17449262738227844,0.0005712529527954757,0.14201408624649048,0.04296307638287544,-0.014093230478465557,0.07598979771137238,-0.07862170785665512,-0.01938720978796482,0.055801376700401306,-0.026732301339507103,-0.013674600049853325,0.051025934517383575,0.00792665220797062,-0.14289574325084686,0.022941164672374725,-0.1038193479180336,-0.03378944471478462,-0.05298750475049019,0.09039824455976486,0.04837571829557419,-0.06494233757257462,-0.14925768971443176,0.015344129875302315,-0.08271805197000504,0.029082242399454117,-0.04234946891665459,0.08439777046442032,0.011856673285365105,0.09760382026433945,0.08309117704629898,-0.045478373765945435,0.1410837471485138,-0.017510641366243362,-0.22694005072116852,0.08483820408582687,0.009065045975148678,-0.19596534967422485,-0.08464422076940536,0.016237325966358185,-0.05917039141058922,0.05780159309506416,0.05513549968600273,0.008733253926038742,-0.12292221933603287,-0.011202247813344002,0.030659589916467667,-0.022913489490747452,0.052020031958818436,0.043835464864969254,-0.035105135291814804,-0.06783699989318848,-0.0009047661442309618,-0.028866281732916832,-0.11532080918550491,-0.0322355292737484,-0.011503060348331928,-0.01817525364458561,-0.06265255063772202,-0.033555835485458374,0.01551132183521986,0.057898443192243576,-0.11223454773426056,0.07129872590303421,0.015172484330832958,-0.011347312480211258,-0.008932441473007202,0.00553472526371479,0.022185062989592552,-0.05930432304739952,0.0858505591750145,0.14611190557479858,-0.11572358757257462,-0.02441318891942501,-0.12871204316616058,0.024701355025172234,0.04448932781815529,-0.09009753912687302,-0.17626972496509552,0.02114637941122055,-0.026742849498987198,0.07859014719724655,-0.045899346470832825,0.00762950861826539,0.05233100429177284,-0.029733402654528618,-0.0525754876434803,-0.05817622318863869,-0.20830784738063812,-0.04350358992815018,-0.023989230394363403,-0.011853808537125587,-0.2240947037935257,-0.03162970766425133,-0.16183024644851685,0.08109919726848602,-0.06491948664188385,-0.026791874319314957,-0.0321328341960907,-0.10533653944730759,-0.024007879197597504,-0.020385421812534332,-0.025323493406176567,-0.024226855486631393,-0.1146443635225296,0.17452482879161835,0.2453649789094925,0.0541490837931633,0.017241304740309715,-0.11946751177310944,-0.11145532876253128,-0.17555604875087738,0.1952071636915207,0.014278159476816654,-0.01421422604471445,-0.01036034431308508,-0.08729015290737152,-0.049943793565034866,-0.04578908160328865,-0.1440543234348297,-0.07586909830570221,0.026523659005761147,-0.150406152009964,0.11596255749464035,0.09773484617471695,0.0023487384896725416,0.024377336725592613,0.020156582817435265,0.05203654244542122,-0.11097516119480133,-0.07070358842611313,-0.013994858600199223,-0.08273202180862427,-0.07007759064435959,-0.011804035864770412,0.04902331903576851,-0.06249431148171425,-0.08972185105085373,0.06401075422763824,0.06843260675668716,-0.02648451179265976,0.008888502605259418,-0.06350784748792648,0.004801001865416765,-0.13557127118110657,-0.0796305239200592,-0.051519665867090225,-0.05853169038891792,0.0893678069114685,0.1048525869846344,-0.07599496841430664,-0.09808608889579773,0.026640480384230614,0.028779298067092896,0.05119191110134125,-0.07596199214458466,0.0907163992524147,-0.015823280438780785,0.05283538997173309,-0.017702147364616394,0.04083926975727081,0.040039077401161194,-0.11604893952608109,0.02526708133518696,-0.07460024952888489,-0.12887151539325714,-0.10207842290401459,0.0017411181470379233,0.13332226872444153,-0.10080239176750183,-0.09342170506715775,0.016763830557465553,-0.019582459703087807,0.025227949023246765,0.052697885781526566,0.03346124663949013,-0.025386739522218704,0.0141410231590271,-0.12518377602100372,0.0002650817623361945,0.05777112394571304,0.015683837234973907,-0.053373631089925766,-0.11794140189886093,-0.042749371379613876,0.07280314713716507,0.01286798994988203,0.13153478503227234,-0.04461713135242462,-0.033468082547187805,0.048054274171590805,-0.07024316489696503,0.05582355335354805,0.0020558489486575127,-0.08155887573957443,0.04473593458533287,0.023873481899499893,-0.012984023429453373,-0.05096403881907463,0.06812597811222076,0.0444701686501503,0.1062941625714302,-0.09991340339183807,0.04939981549978256,0.06316425651311874,0.011196820065379143,-0.053891975432634354,0.07928530871868134,-0.013342010788619518,0.051116153597831726,-0.049825914204120636,0.05079013854265213,0.036833226680755615,0.10085894912481308,-0.0033012954518198967,0.06841527670621872,0.0558878555893898,0.012408318929374218,0.11960567533969879,-0.0918644368648529,0.055290937423706055,-0.09143367409706116,0.08195585757493973,-0.19282284379005432,-0.06144145131111145,-0.044689737260341644,-0.12427354604005814,-0.06831523030996323,-0.01782034896314144,0.03750760480761528,0.022536251693964005,-0.07272493839263916,-0.1433524340391159,-0.10297375172376633,-0.024821611121296883,-0.02672462910413742,0.0009498416911810637,0.04573674499988556,-0.194612517952919,0.0015859316335991025,0.03479341045022011,0.06549309939146042,-0.05477740243077278,-0.05144130066037178,0.10735056549310684,0.05516974627971649,-0.024428006261587143,0.13243696093559265,-0.051372624933719635,-0.02759198285639286,0.059252120554447174,-0.08539501577615738,0.09866008162498474,-0.008289321325719357,0.0057830954901874065,0.05438213422894478,0.24685245752334595,0.11897594481706619,-0.011678868904709816,-0.12756675481796265,-0.02818935737013817,-0.10588644444942474,-0.030568700283765793,0.08015500754117966,-0.017069345340132713,-0.030999314039945602,0.03833237290382385,-0.12453043460845947,0.028065988793969154,-0.027239954099059105,0.028146328404545784,-0.004827608820050955,-0.012121378444135189,-0.0019341562874615192,-0.027592044323682785,-0.010990611277520657,0.05104083567857742,-0.06791459769010544,0.005540693178772926,0.010810835286974907,-0.033585384488105774,0.013540535233914852,-0.021275954321026802,0.002948455512523651,-0.02586427330970764,0.056045014411211014,0.07479964941740036,-0.1267067790031433,0.03701424598693848,0.08282305300235748,0.09871402382850647,-0.053648583590984344,0.010587206110358238,0.044645246118307114,-0.005266323685646057,-0.057964544743299484,-0.11913852393627167,-0.05945775285363197,0.044963229447603226,0.0005164532922208309,-0.0534297414124012,-0.007383052259683609,-0.04189075902104378,0.1016511470079422,0.08517742902040482,0.11794622987508774,-0.05689798295497894,-0.0365980826318264,-0.14776049554347992,-0.03284520283341408,-0.03826044872403145,-0.071070596575737,0.012928754091262817,-0.06462400406599045,0.005100488197058439,-0.05754230171442032,-0.0715646743774414,-0.05457276105880737,0.007206108421087265,0.023272015154361725,0.043053556233644485,0.010668809525668621,0.014396746642887592,0.1351230889558792,0.136127769947052,-0.024143101647496223,0.14608362317085266,0.10095296055078506,-0.03449756279587746,-0.05743294209241867,-0.06692343950271606,0.02410638891160488,-0.022363852709531784,0.01850455440580845,0.010886037722229958,0.010747070424258709,-0.08454539626836777,0.13022704422473907,0.11750461161136627,0.025059672072529793,0.13027998805046082,0.012500871904194355,0.02250080555677414,-0.0872141644358635,0.09149163961410522,0.006582395173609257,-0.17260906100273132,-0.07356267422437668,0.1260252147912979,-0.0111034931614995,-0.1332523375749588,0.01825737953186035,0.04371614009141922,-0.10750049352645874,0.042238179594278336,0.1278316229581833,0.10538680851459503,0.06586036831140518,-0.02688262052834034,0.13059347867965698,0.008250879123806953,0.08803461492061615,-0.04948695749044418,-0.007556857541203499,0.17406919598579407,0.14165610074996948,-0.06243624538183212,-0.022694142535328865,0.05470651015639305,0.06582223623991013,0.11552614718675613,-0.13686327636241913,0.047067925333976746,0.0800837054848671,0.01675761677324772,0.09919857978820801,-0.09170358628034592,-0.01284620352089405,-0.10099457949399948,0.07942526042461395,0.07481640577316284,0.02466212585568428,0.022989701479673386,0.04694635421037674,0.011120127514004707,0.12494073063135147,-0.01597626693546772,-0.019386783242225647,0.026561878621578217,-0.006699098739773035,0.009348670020699501,-0.040391482412815094,0.06256403774023056,-0.09041246026754379,-0.1284807324409485,0.15237592160701752,-0.052611928433179855,-0.010788758285343647,0.0766560509800911,0.11759258061647415,-0.08420788496732712,-0.030558716505765915,-0.0195842906832695,-0.0530472993850708,0.010175397619605064,0.04804234951734543,-0.11014419049024582,-0.032168298959732056,0.06591782718896866,0.05120572820305824,-0.02474065311253071,-0.05932223051786423,-0.0218177642673254,0.08761389553546906,0.07813456654548645,0.027339933440089226,-0.055627934634685516,-0.008626674301922321,-0.018342360854148865,-0.09909680485725403,-0.05684711039066315,0.0022318263072520494,-0.013301774859428406,-0.07768440246582031,0.007261877413839102,0.14128777384757996,0.013930616900324821,0.076616071164608,0.03645966947078705,0.04252474755048752,-0.03638610616326332,0.1562148928642273,0.0029204650782048702,0.023933665826916695,0.017165716737508774,0.0865800529718399,-0.08278163522481918,-0.021677156910300255,-0.004100652411580086,0.01625981740653515,-0.06665358692407608,0.1684287041425705,0.1313903033733368,0.04137619584798813,0.010423083789646626,-0.033703021705150604,0.05799253657460213,0.11889544874429703,0.03339694067835808,-0.004583192989230156,0.11755820363759995,0.1183551475405693,0.01856250688433647,0.12847131490707397,-0.02535034902393818,0.05173691734671593,-0.03836342319846153,-0.09369257092475891,-0.04338225722312927,-0.08179096132516861,0.06685304641723633,-0.0546749010682106,0.00824799109250307,-0.03899337351322174,0.0011758781038224697,-0.07860741764307022,-0.1565266251564026,0.08173251152038574,0.09547348320484161,-0.0841444656252861,0.052347585558891296,-0.1637689620256424,0.07843935489654541,-0.03694406524300575,-0.002918524667620659,0.056595541536808014,-0.17686709761619568,0.006698351353406906,-0.0399111844599247,-0.02383633889257908,-0.14216825366020203,-0.023013046011328697,0.00029552000341936946,0.06225012242794037,-0.04431811347603798,-0.0009889239445328712,-0.11292882263660431,-0.009966270998120308,-0.09061113744974136,-0.021472308784723282,0.00323675200343132,-0.09358367323875427,0.00782817043364048,0.06811419129371643,0.06030009686946869,-0.027365490794181824,0.14871583878993988,0.05717122182250023,0.011804109439253807,-0.15478862822055817,-0.041482970118522644,-0.07327856123447418,-0.012716090306639671,-0.07166609913110733,-0.005349704995751381,-0.0032915386836975813,0.055814266204833984,-0.088051438331604,-0.06509873270988464,-0.03217478096485138,0.0832115039229393,0.026677720248699188,0.05423509702086449,-0.045557793229818344,-0.027327971532940865,-0.04791175201535225,-0.042562391608953476,-0.04460600018501282,-0.058983542025089264,0.046280164271593094,-0.008017790503799915,-0.029488995671272278,-0.06733857095241547,-0.03865279257297516,0.03567596897482872,0.11754463613033295,-0.09446997940540314,-0.05798696354031563,-0.04620487987995148,-0.04327256605029106,-0.05325229838490486,-0.1007763147354126,0.021699700504541397,0.014872961677610874,0.058314669877290726,-0.037408072501420975,0.13602015376091003,0.13248476386070251,-0.06981542706489563,0.060788340866565704,-1.3526287148124538e-05,-0.0006234496249817312,0.026651175692677498,0.07776349037885666,0.04843888431787491,0.06058011204004288,-0.0404171347618103,0.046222712844610214,0.17411307990550995,-0.08676594495773315,0.052905142307281494,-0.12125355005264282,0.15974099934101105,-0.009294270537793636,0.06675158441066742,0.12438761442899704,0.050833724439144135,0.06828922033309937,-0.10225027799606323,0.07290149480104446,0.009300976060330868,0.06747519224882126,-0.03417527675628662,0.038635749369859695,0.1418997049331665,-0.011962157674133778,0.05170338600873947,0.09488686919212341,-0.009185955859720707,0.02121151052415371,0.050729427486658096,-0.07810889929533005,-0.13199996948242188,0.016038868576288223,-0.021452486515045166,-0.14884671568870544,-0.10980452597141266,0.005115242674946785,0.08111736178398132,0.02923644706606865,0.08721288293600082,0.008182822726666927,0.056881051510572433,-0.008768545463681221,-0.02141145057976246,-0.0027845066506415606,-0.04243786260485649,-0.06509864330291748,-0.050080299377441406,-0.10905562341213226,0.0014194708783179522,-0.06510944664478302,-0.03232380375266075,-0.030113430693745613,0.04911881685256958,-0.10773450881242752,0.04078173637390137,0.07211081683635712,-0.022678162902593613,0.09764042496681213,-0.020446820184588432,-0.07516954094171524,0.036753032356500626,-0.05427532643079758,-0.18586263060569763,-0.00032426812686026096,-0.0789867416024208,0.12521502375602722,0.00523150572553277,-0.08792147785425186,0.09115683287382126,0.032550398260354996,0.021754154935479164,-0.05696847662329674,-0.00042063242290169,0.09900154173374176,0.19167892634868622,0.059497103095054626,0.047286197543144226,0.0097421919927001,0.11089776456356049,0.03330668807029724,-0.13002371788024902,0.0013073690934106708,0.04257817566394806,-0.019878583028912544,0.013956412672996521,0.054393548518419266,-0.0004135269555263221,-0.017513670027256012,-0.0697091668844223,-0.0951576754450798,-0.0442889928817749,-0.10028547048568726,0.10152798146009445,-0.16613465547561646,0.09750302135944366,0.010681873187422752,0.10646168887615204,-0.04537806659936905,0.12852273881435394,0.09661709517240524,-0.0917510986328125,-0.04677901789546013,-0.019055241718888283,0.0760936364531517,-0.07929843664169312,-0.03066406026482582,-0.03328751027584076,0.05094754323363304],"yaxis":"y","type":"scattergl"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x0"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"x1"}},"legend":{"tracegroupgap":0,"itemsizing":"constant"},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('eef5afff-a848-44dc-85b0-88b925cc316c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


Five words that I found interpretable and interesting in this visualization are "apparently," "fox," "reportedly," "radical," and "21wire," all of which reside towards the left side of the visualization. "fox" and "21wire" most likely refers to Fox News and 21st Century Wire, which are both news sources that have been criticized for spreading propoganda and false or exhaggerated information. "apparently" and "reportedly" also make sense to me because these are words that are often used by writers when they can't be sure about the information; these words allow them to detach themselves from involvement. "radical" also makes sense as a fake news word because a lot of fake news articles tend to attack "radical leftists."


```python
from google.colab import drive
drive.mount('/content/drive')
```


```python

```
