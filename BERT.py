### import package and create functions

import re
import pandas as pd
import numpy as np
import json
import gzip
from random import sample
import os
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sn

pd.options.mode.chained_assignment = None
reviews_large = pd.read_csv ( 'D:\MSBA\cybersecurity\Merged_Reviews.csv' )
reviews = reviews_large.sample ( n=2000 )

def combined_features(row):
	return row['reviewHeading'] + " " + row['reviewText']
reviews["combined_features"] = reviews.apply ( combined_features, axis=1 )

# def parse(path):
#     g = gzip.open ( path, 'rb' )
#     for l in g:
#         yield json.loads ( l )
#
#
# def getDF(path):
#     i = 0
#     df = {}
#     for d in parse ( path ):
#         df[i] = d
#         i += 1
#     return pd.DataFrame.from_dict ( df, orient='index' )
#
#
# # convert to dataframe
#
# df = getDF ( "D:\MSBA\cybersecurity\data\Home_and_Kitchen_5.json.gz" )
# df.info ()
#
#
#
# # label fake reviews: disclaimer detection
#
# disclaimer = ["discount for review", "discount to review", "for the purpose of a review", "free for my review",
#               "free for review", "free reviewer's sample", "free sample", "free to review", "Freebie",
#               "in exchange for a review",
#               "in exchange for my honest", "in exchange of a review", "in return for a review", "in return of a review",
#               "product for review", "product for test", "review for product", "review sample", "review unit",
#               "reviewing purposes",
#               "sample for an honest review", "sample for review", "sent this for review", "testing and review purposes",
#               "product sent for review"]
#
# fakereviwews = []
#
# for i in df['reviewText']:
#     if any ( j in str ( i ) for j in disclaimer ):
#         fakereviwews.append ( 1 )
#     #        print(str(i))
#     elif "I bought" in str ( i ):
#         fakereviwews.append ( 0 )
#     else:
#         fakereviwews.append ( "NA" )
#
# # %%
#
# fake_index = [i for i, v in enumerate ( fakereviwews ) if v == 1]
# pre_real_index = [i for i, v in enumerate ( fakereviwews ) if v == 0]
# real_index = sample ( pre_real_index, fakereviwews.count ( 1 ) )
#
# fake = df.iloc[fake_index,:]
# fake.loc[:, 'label'] = 1
#
# real = df.iloc[real_index,:]
# real.loc[:, 'label'] = 0
#
# pre_reviews = pd.concat ( [fake, real] )
# # %%
# reviews = pre_reviews.reset_index ( drop=True )
#
# reviews.drop ( columns="style", inplace=True )
#
# dict = {'asin': 'productID',
#         'overall': 'rate',
#         'summary': 'reviewHeading'}
#
# reviews.rename ( columns=dict,
#                  inplace=True )
#
# reviews.loc[:, 'rate'] = reviews.loc[:, 'rate'].astype ( 'int' )
# #reviews['vote'] = reviews['vote'].astype('float')
#
# reviews.loc[:,'reviewTime'] = pd.to_datetime ( reviews.loc[:,'reviewTime'] )
# #reviews.loc[:,['vote','image']] = reviews.loc[:,['vote','image']].fillna(0)
#
# #reviews['natural_log_vote'] = np.log(reviews['vote']+1)
#
#
# reviews.loc[:, ('reviewerName', 'reviewText', 'reviewHeading')] = reviews.loc[:, ('reviewerName', 'reviewText','reviewHeading')].astype ( 'string',copy=False )
#
# reviews.to_csv ( path_or_buf='D:\MSBA\cybersecurity\Merged_Reviews.csv', index=False )

# for i in reviews['image']:
#
# 	if i != 0:
# 		print(i)
# 		print ( type(i) )
# 		i = 1

X_train, X_test, y_train, y_test = train_test_split ( reviews['combined_features'], reviews['label'],
                                                      stratify=reviews['label'] )
print ( X_train.head ( 4 ) )

# def normalize(data):
# 	normalized = []
# 	for i in data:
# 		i = i.lower ()
# 		# get rid of urls
# 		i = re.sub ( 'https?://\S+|www\.\S+', '', i )
# 		# get rid of non words and extra spaces
# 		i = re.sub ( '\\W', ' ', i )
# 		i = re.sub ( '\n', '', i )
# 		i = re.sub ( ' +', ' ', i )
# 		i = re.sub ( '^ ', '', i )
# 		i = re.sub ( ' $', '', i )
# 		normalized.append ( i )
# 	return normalized
#
#
# X_train = normalize ( X_train )
# X_test = normalize ( X_test )
# vocab_size = 10000
# embedding_dim = 64
# max_length = 256
# trunc_type = 'post'
# padding_type = 'post'
# oov_tok = '<OOV>'
# ## tokenizer = Tokenizer(num_words=max_vocab)
# tokenizer = Tokenizer ( num_words=vocab_size, oov_token=oov_tok )
# tokenizer.fit_on_texts ( X_train )
# X_train = tokenizer.texts_to_sequences ( X_train )
# X_test = tokenizer.texts_to_sequences ( X_test )
#
# X_train = tf.keras.preprocessing.sequence.pad_sequences ( X_train, padding=padding_type, truncating=trunc_type,
#                                                           maxlen=max_length )
# X_test = tf.keras.preprocessing.sequence.pad_sequences ( X_test, padding=padding_type, truncating=trunc_type,
#                                                          maxlen=max_length )
# %%

bert_preprocess = hub.KerasLayer ( "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" )
bert_encoder = hub.KerasLayer ( "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4" )

# %%
# Bert layers
text_input = tf.keras.layers.Input ( shape=(), dtype=tf.string, name='text' )
preprocessed_text = bert_preprocess ( text_input )
outputs = bert_encoder ( preprocessed_text )

# Neural network layers
l = tf.keras.layers.Dropout ( 0.1, name="dropout" ) ( outputs['pooled_output'] )
l = tf.keras.layers.Dense ( 1, activation='sigmoid', name="output" ) ( l )

# Use inputs and outputs to construct a final model
model = tf.keras.Model ( inputs=[text_input], outputs=[l] )

# %%
model.summary ()
# %%
len ( X_train )
# %%
METRICS = [
	tf.keras.metrics.BinaryAccuracy ( name='accuracy' ),
	tf.keras.metrics.Precision ( name='precision' ),
	tf.keras.metrics.Recall ( name='recall' )
]

model.compile ( optimizer='adam',
                loss='binary_crossentropy',
                metrics=METRICS )

# %%
model.fit ( X_train, y_train, epochs=10 )
model.evaluate ( X_test, y_test )
y_predicted = model.predict ( X_test )
y_predicted = y_predicted.flatten ()
y_predicted = np.where ( y_predicted > 0.5, 1, 0 )

# %%

cm = confusion_matrix ( y_test, y_predicted )
sn.heatmap ( cm, annot=True, fmt='d' )
plt.xlabel ( 'Predicted' )
plt.ylabel ( 'Truth' )
plt.show ()
print ( classification_report ( y_test, y_predicted ) )