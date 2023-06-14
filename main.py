import labels as labels
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.api._v2.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.layers import Conv1D,Dropout,MaxPooling1D
dataframe=pd.read_excel('C:/Users/HP/Desktop/Python/Project/Project-132/product_dataset/updated_product_dataset.xlsx')
print(dataframe.head())
dataframe["Emotion"].unique()
encode_emotion={"Neutral":0,"Positive":1,"Negative":2}
dataframe.replace(encode_emotion,inplace=True)
dataframe.head()
training_sentences=[]
training_labels=[]
for i in range(len(dataframe)):
    sentence=dataframe.loc[i,"Text"]
    training_sentences.append(sentence)
    label=dataframe.loc[i,"Emotion"]
    training_labels.append(label)
training_sentences[10]
training_labels[10]
vocab_size=10000
embedding_dim=16
max_length=100
trunc_type='post'
padding_type='post'
oov_tok="<OOV>"
training_size=20000
tokenize=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenize.fit_on_texts(training_sentences)
training_sequences=tokenize.texts_to_sequences(training_sentences)
training_padded=pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
training_padded=np.array(training_padded)
training_labels=np.array(training_labels)
myModel=tf.keras.Sequential([
    Embedding(vocab_size,embedding_dim,input_length=max_length),
    Dropout(0.2),
    Conv1D(filters=256,kernel_size=3,activation="relu"),
    MaxPooling1D(pool_size=2),
    LSTM(128),
    Dense(128,activation="relu"),
    Dropout(0.2),
    Dense(128,activation="relu"),
    Dense(6,activation="softmax")
])
myModel.save("customer_review.h5")
loaded_model=tf.keras.models.load_model("customer_review.h5")
print(myModel)
myModel.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
myModel.summary()
history=myModel.fit(training_padded,training_labels,epochs=5,verbose=True)
print(history)
sentence=["Awsome camera. Worth the price"]
sequences=tokenize.texts_to_sequences(sentence)
print(sequences)
padded=pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
print(padded)
result=myModel.predict(padded)
predict_class=np.argmax(result,axis=1)
for emotion in encode_emotion:
    if encode_emotion[emotion]==label:
        print(f"Sentiment: {emotion}, label: {label}")