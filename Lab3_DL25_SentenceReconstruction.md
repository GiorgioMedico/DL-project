#Deep Learning lab 3

Sentence Reordering Task.
Text input:
Text output:
original  (truth):<start> orcas use echolocation to talk to each other and hunt <end>
shuffled (input): <start> talk echolocation and to to use each hunt orcas other <end>
generate (output):


```python
!pip install datasets > /dev/null #> /dev/null to print only the errors
```


```python
import keras
import tensorflow as tf
import numpy as np

from datasets import load_dataset
from keras.layers import TextVectorization

#Hyperparamenters
VOCAB_SIZE = 10000
MAX_SEQ_LEN=28
MIN_SEQ_LEN=9

ds = load_dataset('generics_kb',trust_remote_code=True)['train']
print("Dataset types:",type(ds))
print("Dataset Structure:\n",ds)
ds = ds.filter(lambda row: len(row['generic_sentence'].split(" "))>=MIN_SEQ_LEN )

corpus = ['<start> ' + row['generic_sentence'].replace(","," <comma>") + ' <end>' for row in ds ]
corpus = np.array(corpus)
tokenizer=TextVectorization( max_tokens=VOCAB_SIZE, standardize="lower_and_strip_punctuation")
tokenizer.adapt(corpus)
sentences = tokenizer( corpus ).numpy()

mask = np.sum( (sentences==1) , axis=1) >= 1  #check if <start> appears more than once for each sentence.
original_data = np.delete( sentences, mask , axis=0)
original_data = [sen for sen in original_data if not(1 in sen) and len(sen) <= MAX_SEQ_LEN]

#original_data = [sen for sen in tokenizer(corpus).numpy() if not(1 in sen) and len(sen)>4 and len(sen)<= 32]


```


```python
# Initial Shuffle of the original_data
shuffled_indices = np.random.permutation(len(original_data))
original_data = np.array(original_data)[shuffled_indices]

class TextDetokenizer:
        def __init__(self, vectorize_layer):
          self.vectorize_layer = vectorize_layer
          vocab = self.vectorize_layer.get_vocabulary()
          self.index_to_word = {index: word for index, word in enumerate(vocab)}

        def __detokenize_tokens(self, tokens):

          def check_token(t):
              if t==3:
                  s="<start>"
              elif t==2:
                  s="<end>"
              elif t==7:
                  s="<comma>"
              else:
                  s=self.index_to_word.get(t, '[UNK]')
              return s

          return ' '.join([ check_token(token) for token in tokens if token != 0])

        def __call__(self, batch_tokens):
             return [self.__detokenize_tokens(tokens) for tokens in batch_tokens]


from keras.utils import Sequence
class DataGenerator(Sequence):
        def __init__(self, data, batch_size=32, shuffle=True, seed=None):

            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed
            self.on_epoch_end()

        def __iter__(self):
          for i in range(len(self)):
            yield self[i]

        def __len__(self):
            return int(np.floor(len(self.data) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            data_batch = np.array([self.data[k] for k in indexes])
            #copy of ordered sequences
            result = np.copy(data_batch)
            #shuffle only the relevant positions for each batch
            for i in range(data_batch.shape[0]):
                np.random.shuffle(data_batch[i,1:data_batch[i].argmin() - 1])

            return data_batch,result

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                if self.seed is not None:
                    np.random.seed(self.seed)
                np.random.shuffle(self.indexes)

detokenizer = TextDetokenizer( tokenizer )

#data split
total_len = len(original_data)
last_train_idx = 220000

print("Total length:", total_len)
print("Train set length: ",last_train_idx)
print("Test set length: ",total_len - last_train_idx)

train_generator = DataGenerator(original_data[:last_train_idx])
test_generator = DataGenerator(original_data[last_train_idx:])

detokenizer = TextDetokenizer(tokenizer)
batch_x, batch_y = train_generator[0]

detokenized_x = detokenizer(batch_x)
detokenized_y = detokenizer(batch_y)


for i in range(3):
    print(f"\nSample {i+1}: from shuffled to original")
    # Token (sequenza di numeri)
    print(f"Tokenized input:", ' '.join(map(str, batch_x[i])))
    print(f"Tokenized output:", ' '.join(map(str, batch_y[i])))
    # Detokenized (frase decodificata)
    print(f"Text input:", detokenized_x[i])
    print(f"Text output:", detokenized_y[i])

```

Let's work on the model.
We'll training and test a transformer


```python
!pip install --upgrade keras-hub keras > /dev/null

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

```


```python
from keras import Model
from keras.layers import Input, Embedding, Dense
from keras_hub.layers import TransformerEncoder, TransformerDecoder
import keras

# Hyperparameters
vocab_size = VOCAB_SIZE
sequence_length=28

intermediate_dim = 128
num_heads=12
embedding_dim=512
num_layers=3

# 3L config is
# intermediate_dim = 64
# num_heads=8
# embedding_dim=128
# num_layers=3

# 1L config is
# intermediate_dim = 64
# num_heads=8
# embedding_dim=64
# num_layers=1



# Modifichiamo le funzioni per incorporare gli embedding layer
def create_encoder_model():
    encoder_inputs = Input(shape=(sequence_length,), name="encoder_input")

    # Embedding layer per l'encoder
    embedding = Embedding(
                 input_dim=vocab_size,
                 output_dim=embedding_dim,
                 name="encoder_embedding"
                )(encoder_inputs)

    # Transformer encoder blocks
    encoder_outputs = embedding
    for _ in range(num_layers):
        encoder_outputs=TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads
            )(encoder_outputs)

    return Model(inputs=encoder_inputs, outputs=encoder_outputs, name="Encoder_Model")


def create_decoder_model():
    seqs_inputs = Input(shape=(sequence_length-1,), name="decoder_input")
    embedding = Embedding(
                  input_dim=vocab_size,
                  output_dim=embedding_dim,
                  name="decoder_embedding"
                )(seqs_inputs)

    enc_outputs = Input(shape=(sequence_length, embedding_dim), name="encoder_output")

    # Transformer dencoder layers
    denc_outputs = embedding

    for _ in range(num_layers):
        dec_outputs=TransformerDecoder(intermediate_dim=intermediate_dim,num_heads=num_heads)(embedding, enc_outputs)

    # Proiezione finale sul vocabolario
    outputs = Dense(vocab_size, activation='softmax')(dec_outputs)

    return Model(inputs=[seqs_inputs, enc_outputs], outputs=outputs, name="Decoder_Model")



# Creazione dei modelli separati
encoder_model = create_encoder_model()
decoder_model = create_decoder_model()


encoder_model.summary()
print()
decoder_model.summary()

```

We have to understand the training process first.

What the model predict is the next token of a sequence.

The output of the decoder is the vocaboulary-shape vector of probabilities.


```python
# Costum train loop of a Combined Keras Model
class SLM(keras.Model):
    def __init__(self, encoder_model, decoder_model, **kwargs):
        super(SLM, self).__init__(name="SLM_Model",**kwargs)
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.loss_tracker = keras.metrics.Mean(name='loss')

    def save_path(self,name):
      return f"/content/drive/MyDrive/DLlab_2425/sentence_reordering_{name}.weights.h5"

    def load_weights(self,):
        self.encoder.load_weights(self.save_path("enc"))
        self.decoder.load_weights(self.save_path("dec"))
        return

    def save_weights(self, ):
        self.encoder.save_weights(self.save_path("enc"))
        self.decoder.save_weights(self.save_path("dec"))
        return

    def train_step(self, data):
        batch_x, batch_y = data

        with tf.GradientTape() as tape:

            # Encoder forward pass
            encoder_outputs = self.encoder(batch_x, training=True)

            # Right shift of the target sentence as input
            decoder_inputs = batch_y[:,:-1]

            # Decoder forward pass
            predictions = self.decoder([decoder_inputs, encoder_outputs], training=True)

            # Compute the loss
            loss = self.compiled_loss(batch_y[:,1:], predictions)

        # Compute the gradients
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update the weigths
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


    def call(self, batch_x, training=False):
      return self.encoder(batch_x, training=training)

    def generate(self, batch_of_shuffled_tokens, training=False, max_length=sequence_length):
        contexts = self.encoder(batch_of_shuffled_tokens, training=training)
        predictions = []
        for i, ctx in enumerate(contexts):
            ctx = tf.expand_dims(ctx, 0)
            end_token = 2
            start_token= tf.constant([[3]]) #start_token value
            dummy_paddings = tf.zeros((1, max_length - 2), dtype=tf.int32) #add paddings to fill the sentence

            output_sequence = tf.concat([start_token, dummy_paddings ], axis=1)

            for step in range(max_length-2):
              preds = self.decoder([output_sequence, ctx], training=training)
              #next token is the last most likely index of the vocabulary
              next_token = tf.argmax(preds[:, step, :] , axis=-1, output_type=tf.int32)

              #output_sequence[step+1]= next_token
              output_sequence = tf.tensor_scatter_nd_update(
                output_sequence, [[0, step + 1]], [next_token[0]]  # Use step + 1 for correct index
              )

              if next_token[0].numpy() == end_token:
                break

            predictions.append(output_sequence)

        results = tf.concat(predictions, axis=0).numpy()
        return results

    def generate_vec(self, batch_of_shuffled_tokens, training=False, max_length=sequence_length):
        contexts = self.encoder(batch_of_shuffled_tokens, training=training)

        batch_size = tf.shape(contexts)[0]
        end_token = 2

        # we initialize start+puddings for each sequence for each context
        start_tokens = tf.fill([batch_size, 1], 3)  # start_token value
        dummy_paddings = tf.zeros([batch_size, max_length - 2], dtype=tf.int32)
        output_sequences = tf.concat([start_tokens, dummy_paddings], axis=1)

        # tracking of active_sequences
        active_sequences = tf.ones([batch_size], dtype=tf.bool)

        def loop_body(step, output_sequences, active_sequences):
            preds = self.decoder([output_sequences, contexts], training=training)
            next_tokens = tf.argmax(preds[:, step, :], axis=-1, output_type=tf.int32)
            indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], step + 1)], axis=1)
            output_sequences = tf.tensor_scatter_nd_update(
                output_sequences, indices, next_tokens
            )
            is_end_token = tf.equal(next_tokens, end_token)
            active_sequences = active_sequences & ~is_end_token
            return step + 1, output_sequences, active_sequences


        # Use tf.while_loop instead of Python for loop
        _, output_sequences, _ = tf.while_loop(
            cond= lambda step, *_: step < max_length - 2 and tf.reduce_any(active_sequences),
            body= loop_body,
            loop_vars=[0, output_sequences, active_sequences]
        )

        return output_sequences.numpy()

    @property
    def metrics(self):
        return [self.loss_tracker]




# Creazione e compilazione del modello combinato
model = SLM(encoder_model, decoder_model)


from keras.losses import sparse_categorical_crossentropy

def custom_vocab_sparsecatcrossentropy(y_true, y_pred):
    loss = sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=custom_vocab_sparsecatcrossentropy
)
model.summary()

```


```python
from difflib import SequenceMatcher
class ScoreCallback(keras.callbacks.Callback):
    def __init__(self, epoch_interval=None, valid_set=test_generator):
        self.epoch_interval = epoch_interval
        self.valid_set = valid_set

    def compute_score_on_sentence(self, s,p):
        match = SequenceMatcher(None, s, p).find_longest_match()
        return (match.size/max(len(p),len(s)))

    def score_fun(self, sentences,predictions):
      scores = map(lambda s, p: self.compute_score_on_sentence(s, p), sentences, predictions)
      return np.mean(list(scores))

    def on_epoch_end(self, epoch, logs=None):
        #if self.epoch_interval and epoch % self.epoch_interval == 0:
        if self.epoch_interval and (epoch + 1) % self.epoch_interval == 0:
            cum_scores=[]
            self.model.save_weights()
            print("\nSaved models, computing scores..")

            #take a random batch form valid_set
            batch_x, batch_y = self.valid_set[np.random.randint(len(self.valid_set))]

            # for batch_x, batch_y in self.valid_set:
            preds  = self.model.generate(batch_x)
            generated = detokenizer(preds)
            originals = detokenizer(batch_y)

            print("\nOriginal:  ", originals[0])
            print("Shuffled:  ", detokenizer(batch_x[None, 0])[0])
            print("Generated: ", generated[0])
            print("Got score: ", self.compute_score_on_sentence(originals[0], generated[0]))

            score = self.score_fun(originals, originals)
            print(f"Batch score: {np.mean(np.array(score))}\n",)



early_stopping = keras.callbacks.EarlyStopping(
    monitor="loss",
    restore_best_weights=True,
    start_from_epoch=0,
    patience=3
)


model.fit(
    train_generator,
    epochs=100,
    callbacks=[ScoreCallback(10),early_stopping]
)

```


```python



model.save_weights()
```


```python
from difflib import SequenceMatcher

#Baseline test
def compute_score_on_sentence(s,p):
    matches = SequenceMatcher(None, s, p).find_longest_match()
    return (matches.size/max(len(p),len(s)))

from difflib import SequenceMatcher
def compute_score_on_batch(seq,pred):
    scores = map(lambda s, p: compute_score_on_sentence(s, p), seq, pred)
    return np.mean(list(scores))


#compute the baseline
cum_scores=[]
from tqdm import tqdm

total_iters = len(test_generator)
for i, (batch_x, batch_y) in enumerate(tqdm(test_generator, desc="Processing batches", unit="batch")):
    #completely random shuffle
    generated = detokenizer(batch_x)
    originals = detokenizer(batch_y)
    score_value = compute_score_on_batch(originals, generated)
    cum_scores.append(score_value)


cum_scores = np.array(cum_scores)
baseline = np.mean(cum_scores)
print("\nBaseline Score: ",baseline)
print("Acceptable Score: ",baseline + 3 * np.std(cum_scores))

```


```python
#final test

VERBOSE=False
model.load_weights()
cum_scores=[]
for i, (batch_x, batch_y) in enumerate(tqdm(test_generator, desc="Final Test", unit="batch")):
    pred_tokens  = model.generate_vec(batch_x)
    generated = detokenizer(pred_tokens)
    originals = detokenizer(batch_y)
    score_value = compute_score_on_batch(originals, generated)
    cum_scores.append(score_value)
    if VERBOSE:
      print("\nOriginal:  ", originals[0])
      print("Shuffled: ", detokenizer(batch_x[None, 0])[0])
      print("Generated: ", generated[0])
      print("Got score: ",score_value)

print("Matching Score: ",np.mean(np.array(cum_scores)))
```


```python
#Previous model final test
VERBOSE=False
model.load_weights()
cum_scores=[]
for i, (batch_x, batch_y) in enumerate(tqdm(test_generator, desc="Final Test", unit="batch")):
    pred_tokens  = model.generate_vec(batch_x)
    generated = detokenizer(pred_tokens)
    originals = detokenizer(batch_y)
    score_value = compute_score_on_batch(originals, generated)
    cum_scores.append(score_value)
    if VERBOSE:
      print("\nOriginal:  ", originals[0])
      print("Shuffled: ", detokenizer(batch_x[None, 0])[0])
      print("Generated: ", generated[0])
      print("Got score: ",score_value)

print("Matching Score: ",np.mean(np.array(cum_scores)))
```
