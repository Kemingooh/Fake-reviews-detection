### import package and create functions

import re
import pandas as pd
import json
import gzip
from random import sample
import os
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sn
#from tensorflow.python.keras.preprocessing.text import Tokenizer
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pd.options.mode.chained_assignment = None
reviews_large = pd.read_csv ( 'D:\MSBA\Cybersecurity\Merged_Reviews.csv' )
reviews = reviews_large.sample ( n=2000 )

def combined_features(row):
	return str(row['reviewerName']) + " " + row['reviewHeading'] + " " + row['reviewText']
reviews["combined_features"] = reviews.apply ( combined_features, axis=1 )

X_train, X_test, y_train, y_test = train_test_split ( reviews['combined_features'], reviews['label'],stratify=reviews['label'] )

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
# X_train = normalize ( X_train )
# X_test = normalize ( X_test )
# vocab_size = 10000
# embedding_dim = 64
# max_length = 256
# trunc_type = 'post'
# padding_type = 'post'
# oov_tok = '<OOV>'
# tokenizer = Tokenizer(num_words=vocab_size)
# tokenizer = Tokenizer ( num_words=vocab_size, oov_token=oov_tok )
# tokenizer.fit_on_texts ( X_train )
# X_train = tokenizer.texts_to_sequences ( X_train )
# X_test = tokenizer.texts_to_sequences ( X_test )

# X_train = tf.keras.preprocessing.sequence.pad_sequences ( X_train, padding=padding_type, truncating=trunc_type,
#                                                           maxlen=max_length )
# X_test = tf.keras.preprocessing.sequence.pad_sequences ( X_test, padding=padding_type, truncating=trunc_type,
#                                                          maxlen=max_length )


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Bert layers
text_input = tf.keras.layers.Input ( shape=(), dtype=tf.string, name='text' )
preprocessed_text = bert_preprocess ( text_input )
outputs = bert_encoder ( preprocessed_text )

# Neural network layers
l = tf.keras.layers.Dropout ( 0.1, name="dropout" ) ( outputs['pooled_output'] )
l = tf.keras.layers.Dense ( 1, activation='sigmoid', name="output" ) ( l )

# Use inputs and outputs to construct a final model
model = tf.keras.Model ( inputs=[text_input], outputs=[l] )

model.summary ()


METRICS = [
	tf.keras.metrics.BinaryAccuracy ( name='accuracy' ),
	tf.keras.metrics.Precision ( name='precision' ),
	tf.keras.metrics.Recall ( name='recall' )
]

model.compile ( optimizer='adam',
                loss='binary_crossentropy',
                metrics=METRICS )


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

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
prediction = dict ()
prediction['BERT'] = y_predicted
cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items ():
	false_positive_rate, true_positive_rate, thresholds = roc_curve ( y_test, predicted )
	roc_auc = auc ( false_positive_rate, true_positive_rate )
	plt.plot ( false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc) )
	cmp += 1

plt.title ( 'Classifiers comparison with ROC' )
plt.legend ( loc='lower right' )
plt.plot ( [0, 1], [0, 1], 'r--' )
plt.xlim ( [-0.1, 1.2] )
plt.ylim ( [-0.1, 1.2] )
plt.ylabel ( 'True Positive Rate' )
plt.xlabel ( 'False Positive Rate' )
plt.show ()