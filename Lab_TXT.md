# Text and Transformers lab
In this lecture: a bit of text processing, loading text in keras, classification with 1DConvnet & transformers

Let's start by reviewing a bit of text processing

Unlike Images, Text has quite a few criticalities. Just to name a few:

- High Dimensionality: Text data, especially when tokenized into individual words or n-grams, can lead to a very high-dimensional feature space. This can make models computationally expensive and increase the risk of overfitting.

- Sparsity: Most text documents will only contain a tiny fraction of the words in a language, resulting in many zeros in the feature representation. This sparsity can make certain modeling techniques inefficient or infeasible.

- Ambiguity and Polysemy: Many words in languages have multiple meanings based on context. For example, the word "bank" can mean the side of a river or a financial institution.

- Synonymy: Different words can have similar meanings, like "big" and "large". This poses a challenge in identifying the true intent or sentiment behind texts.

- Complex Dependencies: The meaning of a word can depend on its surrounding words, or even words much earlier in a text. Capturing long-term dependencies can be challenging.

- Noisy Data: Text data, especially from sources like social media, can be noisy. They may contain typos, slang, non-standard grammar, or emoticons.

- Handling of Out-of-Vocabulary (OOV) Words: In real-world applications, it's common to encounter words not seen during training. Handling OOV words is challenging.

- Cultural and Temporal Dynamics: The way language is used can change based on cultural or temporal contexts, making it hard to generalize models across different cultures or time periods.

- Multilingual Challenges: Building models that work across multiple languages or even dialects can be challenging, especially when resources for certain languages are scarce.

### some examples of regex


```python
import re
text = "Hello, World!"
match = re.search("World", text)
if match:
    print("Found:", match.group())
```


```python
text = "I am Free."
match = re.search("[FT]ree", text)
if match:
    print("Found:", match.group())
```


```python
text = "Order number: 12345"
match = re.search("\d+", text)  # + indicates one or more
if match:
    print("Found:", match.group())
```


```python
text = "Hello"
if re.match("^Hello", text):  # ^ matches the start
    print("Starts with 'Hello'")
if re.search("Hello$", text):  # $ matches the end
    print("Ends with 'Hello'")
```


```python
text = "apple,banana,orange"
fruits = re.split(",", text)
print(fruits)  # ['apple', 'banana', 'orange']
```


```python
text = "HELLO"
match = re.search("hello", text, re.IGNORECASE)
if match:
    print("Found:", match.group())
```


```python
text = "this is my email fabio.merizzi@unibo.it"
match = re.search("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", text)
if match:
    print("Found email:", match.group())
```

### Some basic ideas about text preprocessing
Reducing the high dimensionality of text

A simple idea is to lowercase every word, casing has little meaning


```python
text = "Hello here is some text, BYE"
# Lowercasing
lowercased_text = text.lower()
print("Lowercased Text:", lowercased_text)
# Tokenization
tokens = lowercased_text.split()
print("Tokens:", tokens)
```

Punctuations as well


```python
#remove punctuations
import string
text = "Hello, World! who's speaking? Ah, it's you!"
clean_text = text.translate(str.maketrans('', '', string.punctuation))
print("Cleaned Text:", clean_text)
```


```python
#some words have less meaning
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
text = "This is an example of removing stopwords from a sentence."
tokens = text.split()
filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
print("Filtered Tokens:", filtered_tokens)
```

Some words have very similiar meaning but different coniugations, these may impair the classification, for addressing this problem we can employ Stemming and Lemming. 

Stemming truncates words by chopping off the ends of words using heuristic processes.


```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
text = "running runner runs run"
tokens = text.split()
stemmed_tokens = [ps.stem(token) for token in tokens]
print("Stemmed Tokens:", stemmed_tokens)
```

Lemming reduces words to their base or root form by considering the dictionary form of the word


```python
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
text = "geese mice swimming swam"
tokens = text.split()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized Tokens:", lemmatized_tokens)
```

### Encoding Text
transforming text into numbers

#### One hot encoding
Practical, but only with very low cardinality


```python
from sklearn.preprocessing import OneHotEncoder

words = [["hello"], ["this"], ["is"], ["the"], ["machine"], ["learning"], ["course"]]

encoder = OneHotEncoder(sparse_output=False)
encoded_words = encoder.fit_transform(words)

print("One-Hot Encoded Words:\n", encoded_words)
```

#### Label Encoding, great for transforming labels


```python
from sklearn.preprocessing import LabelEncoder

labels = ["cat", "dog", "bird", "cat", "bird"]
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
print(encoded_labels) 
```

#### Bag of Words (BoW - TF) Encoding is essentially a histogram of word frequencies in a text document. For each document or text sample, count the number of occurrences of each word and represent the document as this count vector.

 When to use it:

    Document classification: BoW is popular for tasks like email spam classification or sentiment analysis, where the occurrence of certain words can be a strong indicator of the class.
    Texts with different lengths: Since BoW leads to fixed-length vectors (the size of the vocabulary), it's useful when dealing with texts of varying lengths.
    Simple models: For models like Naive Bayes, BoW can be very effective.

Cons:

    Loses all information about word order. "This is good, not bad" and "This is bad, not good" would have the same representation.
    Like one-hot encoding, BoW can also lead to high-dimensional data with large vocabularies.
    Doesn't capture semantic relationships between words.


```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Hello there this is the machine learning course, hello again",
    "Welcome to the machine learning class."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Representation:\n", X.toarray())
```

#### TF-IDF
TF-IDF is a statistical measure that evaluates the importance of a word in a document, relative to a collection of documents (often called a corpus). It's composed of two terms:

Term Frequency (TF): Represents how often a term appears in a given document.

$$TF(t,d)= \frac{Number \, of \, times \, term \,t \,appears \,in \,document \,d}{Total\, number \,of \,terms \,in\, document\, d}$$


Inverse Document Frequency (IDF): Represents the significance of the term across all documents in the corpus. It assigns more weight to terms that are rare across documents.

$$IDF(t)= log⁡(\frac{Total\, number\, of\, documents}{Number\, of \,documents \,containing\, term\, t})$$

The TF-IDF value for a term in a document is then:
$$TF-IDF(t,d)=TF(t,d)×IDF(t)$$

Words with high TF-IDF scores are those that are important in a given document relative to the entire corpus.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I love machine learning.",
    "Machine learning is challenging.",
    "Python is a popular language for machine learning.",
]

# Initialize a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF values
tfidf_matrix = vectorizer.fit_transform(docs)

# Feature names (words in the vocabulary)
features = vectorizer.get_feature_names_out()

```


```python
features
```


```python
tfidf_matrix.toarray()
```

#### What we really want: dense word embeddings

The encodings methods mentioned above have two main drawbacks we need to solve. 1) The very high dimensionality of the embedding 2) similar words/concept are placed randomly in the encoding space.

We well see in the following implementations how to solve both this issues via a neural embedding

### Loading text from file to keras dataset

IMDB movie reviews, one of the most common dataset for mlp


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers
from keras import ops
```


```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```


```python
!cat aclImdb/train/pos/6248_7.txt
```


```python
!rm -r aclImdb/train/unsup
```


```python
batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.5,
    subset="training",
    seed=1337,
)
raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.5,
    subset="validation",
    seed=1337,
)
raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

```


```python
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
```

### Prepare text data in the keras framework


```python
import string
import re
#Define custom a preprocessing procedure
def custom_preprocessing(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    no_punctuation = tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]", "")
    return no_punctuation
```


```python
string.punctuation
```


```python
#Let's write our text vectorization layer 
# it will preprocess and map our words into a 
vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_preprocessing,
    max_tokens=20000,
    output_mode="int",
    output_sequence_length=500,
)
# Output can be int, multi_hot or tf_idf

# Now that the vectorize_layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)
```


```python
a,b = next(iter(raw_train_ds))
```


```python
a
```


```python
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
```


```python
a,b = next(iter(train_ds))
```


```python
a[0]
```


```python
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
```


```python
# A integer input for vocab indices.
inputs = keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(20000,128)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
```


```python
model.summary()
```


```python
epochs = 15

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
```


```python
model.evaluate(test_ds)
```

# Let's use a Transformer
Transformer are based on the idea of attention

Understanding multi-head attention: The main idea behind multi-head attention is to split the attention mechanism into multiple "heads." By doing this, the model can capture various aspects of the data in parallel.

The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value.

$$Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt(d_k)})*V$$

The interaction between Queries, Keys, and Values happens as follows in the attention mechanism:

- Scoring: Each query vector is scored against all key vectors using a dot product, which measures how much each element (represented by a query) should attend to every other element (represented by keys).

- Scaling: The scores are then scaled down by the square root of the dimension of the key vectors to help stabilize the gradients during learning.
- Softmax: A softmax function is applied to the scaled scores, converting them into probabilities that sum to one. This softmaxed score represents the attention weights.
- Application: These attention weights are then used to create a weighted sum of the value vectors, which forms the output for each position in the input sequence.

Practically, values and keys are usually the same. In the case of self attention, Queries, Keys and Values are all the same. 

So where is the learning actually happening? In the dense layers (Linear) before the computation of attention.


```python
# A possible implementation of multi head attention
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_size = model_size
        
        assert model_size % self.num_heads == 0  # Ensure the model size is divisible by number of heads
        
        self.depth = model_size // self.num_heads
        
        self.wq = tf.keras.layers.Dense(model_size)  # Weight matrices for queries
        self.wk = tf.keras.layers.Dense(model_size)  # Weight matrices for keys
        self.wv = tf.keras.layers.Dense(model_size)  # Weight matrices for values

        self.dense = tf.keras.layers.Dense(model_size)  # Final dense layer after concatenation

    def split_heads(self, x, batch_size):
        #Split the last dimension into (num_heads, depth).
        #Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # Linear projection
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # Split heads
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply the softmax is done on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_output = tf.reshape(output, (batch_size, -1, self.model_size))  # (batch_size, seq_len_q, model_size)
        
        output = self.dense(concat_output)  # Pass through the final dense layer
        
        return output

```


```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs) # Multi head attention where Key, Value and Query are all the same
        attn_output = self.dropout1(attn_output) # We add a dropout to reduce overfitting
        out1 = self.layernorm1(inputs + attn_output) # We add a residual connection and layernorm the result 
        ffn_output = self.ffn(out1) # Feedforward network
        ffn_output = self.dropout2(ffn_output) # a second dropout
        return self.layernorm2(out1 + ffn_output) # a second residual connection

```

Transformers need both token embedding (the words) and position embedding (the token order) because the transformer architecture does not inherently process sequential data with an awareness of order or position


```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        # The embedding layer turns positive integers intodense vectors,
        # (Words with similar meaning are close to each other)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        # get the number of tokens 
        maxlen = ops.shape(x)[-1]
        # get all positions in order
        positions = ops.arange(start=0, stop=maxlen, step=1)
        # the the embedded positions
        positions = self.pos_emb(positions)
        # compute the token embeddings
        x = self.token_emb(x)
        # finally return the embedded tokens + the positions 
        return x + positions
```

Beware! this is a simple transformer approach for classification, tasks such as text generation, translation and so on requires a much more complex model. 


```python
vocab_size = 20000  # Only consider the top 20k words
maxlen = 500 # max number of input tokens
embed_dim = 32  # Embedding size for each token
num_heads = 16  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer
```


```python

inputs = layers.Input(shape=(maxlen,)) # the input is a sequence of maxlen tokens (if not long enough is padded with zeros)
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim) # The embedding layer embed tokens and positions
x = embedding_layer(inputs) 
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim) 
x = transformer_block(x) # A transformer block process the data
# What follows is a simple classifier 
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```


```python
model.summary()
```


```python
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_ds, batch_size=32, epochs=10, validation_data=val_ds
)

```


```python
model.evaluate(test_ds)
```
