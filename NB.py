### import package and create functions

import pandas as pd
import numpy as np
import json
import gzip
from random import sample
import os

from sklearn.model_selection import train_test_split
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

pd.options.mode.chained_assignment = None
reviews_large = pd.read_csv ( 'D:\MSBA\cybersecurity\Merged_Reviews.csv' )
reviews = reviews_large.sample ( n=2000 )

def combined_features(row):
	return row['reviewHeading'] + " " + row['reviewText']

reviews["combined_features"] = reviews.apply ( combined_features, axis=1 )

# %%
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

# reviews.to_csv ( path_or_buf='D:\MSBA\cybersecurity\Merged_Reviews.csv', index=False )

# for i in reviews['image']:
#
# 	if i != 0:
# 		print(i)
# 		print ( type(i) )
# 		i = 1
# %%
X_train, X_test, y_train, y_test = train_test_split ( reviews['combined_features'], reviews['label'], test_size=0.2,
                                                      random_state=42 )
print ( X_train.head ( 4 ) )

stemmer = PorterStemmer ()
from nltk.corpus import stopwords
def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append ( stemmer.stem ( item ) )
	return stemmed


def tokenize(text):
	tokens = nltk.word_tokenize ( text )
	# tokens = [word for word in tokens if word not in stopwords.words('english')]
	stems = stem_tokens ( tokens, stemmer )
	return ' '.join ( stems )


intab = string.punctuation
outtab = "                                "
trantab = str.maketrans ( intab, outtab )

# --- Training set

corpus = []
for text in X_train:
	text = text.lower ()
	text = text.translate ( trantab )
	text = tokenize ( text )
	corpus.append ( text )

count_vect = CountVectorizer ()
X_train_counts = count_vect.fit_transform ( corpus )

tfidf_transformer = TfidfTransformer ()
X_train_tfidf = tfidf_transformer.fit_transform ( X_train_counts )

# --- Test set

test_set = []
for text in X_test:
	text = text.lower ()
	text = text.translate ( trantab )
	text = tokenize ( text )
	test_set.append ( text )

X_new_counts = count_vect.transform ( test_set )
X_test_tfidf = tfidf_transformer.transform ( X_new_counts )

from pandas import *

df = DataFrame ( {'Before': X_train, 'After': corpus} )
print ( df.head ( 20 ) )

prediction = dict ()

# %%
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB ().fit ( X_train_tfidf, y_train )
prediction['Multinomial'] = model.predict ( X_test_tfidf )

# %%
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB ().fit ( X_train_tfidf, y_train )
prediction['Bernoulli'] = model.predict ( X_test_tfidf )

# %%
from sklearn import linear_model

logreg = linear_model.LogisticRegression ( C=1e5 )
logreg.fit ( X_train_tfidf, y_train )
prediction['Logistic'] = logreg.predict ( X_test_tfidf )

#%%
# # Import the model we are using
# from sklearn.ensemble import RandomForestRegressor
# # Instantiate model with 100 decision trees
# rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# # Train the model on training data
# rf.fit(X_train_tfidf, y_train)
# prediction['Rf'] = logreg.predict ( X_test_tfidf )

# %%

def formatt(x):
	if x == 'negative':
		return 0
	return 1


vfunc = np.vectorize ( formatt )

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items ():
	false_positive_rate, true_positive_rate, thresholds = roc_curve ( y_test.map ( formatt ), vfunc ( predicted ) )
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

# %%
print ( metrics.classification_report ( y_test, prediction['Logistic'], target_names=["positive", "negative"] ) )
print ( metrics.classification_report ( y_test, prediction['Multinomial'], target_names=["positive", "negative"] ) )
print ( metrics.classification_report ( y_test, prediction['Bernoulli'], target_names=["positive", "negative"] ) )
# print ( metrics.classification_report ( y_test, prediction['Rf'], target_names=["positive", "negative"] ) )
