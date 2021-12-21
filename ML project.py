# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:24:57 2021

@author: Vladi
"""

# conda activate tf_2.7

# Machine Learning 1
import re
import os
from datetime import datetime
import time
from packaging import version
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
os.chdir("C:/Users/Vladi/OneDrive/Documents/HEC/ML project")

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 2000)
pd.set_option('max_colwidth', 40)


#---------------------------
# Step 1: Preprocessing data
#---------------------------
#-- Remove (the as)
#-- Select categories (69_000) Take the top 20 categories
#-- Understand the different categories
#-- Set up TF and TFIDF




# Define our sample set for the experiment
create_sample = False
if create_sample == True:
    document_df = pd.read_json("arxiv-metadata-oai-snapshot.json", lines=True)
    
    # See how many categories
    num_cat = document_df['categories'].value_counts()
    #count_of_count = num_cat.value_counts()
    
    # Fetch only top 20 categories
    out_top_20_cats = num_cat[:5].index
    document_df_20 = document_df[document_df.categories.isin(out_top_20_cats )][['id', 'title','abstract', 'categories']]
    
    # Sample our data 
    our_sample = document_df_20.sample(n=round(document_df_20.shape[0]*0.1), random_state=1)
    
    # write a pandas dataframe to zipped CSV file
    compress = False
    if compress == True:
        our_sample.to_csv("sample_20_categories_v2.csv.zip", 
                          index=False,
                          compression="zip")

# Preproccess the data
clean_data = False
if clean_data == True:
    
    # Read our sample data
    sample_data = pd.read_csv("sample_20_categories.csv.zip", compression='zip')
    
    #-------------------------------
    # Raw preprocessing of abstracts
    #-------------------------------
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    import swifter #parallel 'apply'
    
    # Remove numbers and other characters
    def clean_abstract(data):
        raw = re.sub('[^A-Za-z]+', ' ', data)
        raw = re.sub(r'\b\w\b', ' ', raw)
        return re.sub(r'\s{2,}', ' ', raw).strip()
    
    
    def stop_words(data):
        data = data.lower()
        data = data.split()
        data = [word for word in data if not word in stopwords.words('english')]
        return ' '.join(data)
    
    # Get only the categoies we want
    limited_cat = ['cs.CV', 'math.PR', 'astro-ph', 'nucl-th']
    sample_data = sample_data[sample_data['categories'].isin(limited_cat)]
    
    sample_data['abstract_clean'] = sample_data['abstract'].map(clean_abstract)
    sample_data['abstract_tensor'] = sample_data['abstract_clean'].swifter.allow_dask_on_strings(enable=True).apply(stop_words)
    
    sample_data = sample_data[['id', 'title', 'abstract', 'categories', 'abstract_tensor']]
    sample_data.to_csv("4_categories_stopw.csv.zip", 
                          index=False,
                          compression="zip")
    
# Read our sample data
sample_data = pd.read_csv("4_categories_stopw.csv.zip", compression='zip')

# Give each unique category a number, useful for manipulating in tensorflow
unique_categories = pd.DataFrame(sample_data.categories.unique().tolist(), columns = ['categories'])
unique_categories['cat_id'] = unique_categories.index

sample_data = pd.merge(sample_data, unique_categories, on = 'categories', how = 'left')

#-----------------------
# Descriptive statistics
#-----------------------
def get_count(data):
    data = data.split() 
    return len(data)


sample_data['n_words'] =  sample_data['abstract'].map(get_count)
sample_data['n_words'].mean()
fig = px.histogram(sample_data, x="n_words", color="categories")
fig.show()


my_valz = pd.DataFrame(sample_data["categories"].value_counts())
my_valz['Categories'] = my_valz.index
my_valz['values']  = my_valz['categories']
fig = px.bar(my_valz, x='Categories', y='values', text = 'values' )
fig.show()

sample_data.groupby(by=["categories"])['n_words'].describe()


sample_data['abstract'].iloc[5]
sample_data['abstract_tensor'].iloc[5]

#------------------------------------------------------
# NOTE: Problem of plural words like 'wall' and 'walls'
#------------------------------------------------------

# Split into training testing
# 10% is kept for testing on different models
X_pre_train, X_test, y_pre_train, y_test = train_test_split(sample_data[['id', 'title', 'abstract_tensor']], 
                                                    sample_data.categories, 
                                                    test_size=0.1, 
                                                    random_state=42)

# 20% for validation and 70% for training
X_train, X_val, y_train, y_val = train_test_split(X_pre_train, 
                                                    y_pre_train, 
                                                    test_size=0.2, 
                                                    random_state=42)

print(f'''
      Training: {X_train.shape[0]/sample_data.shape[0]*100} 
      Validation: {X_val.shape[0]/sample_data.shape[0]*100}
      Testing: {X_test.shape[0]/sample_data.shape[0]*100}''')



#------------------
# Majority Baseline
#------------------ 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%matplotlib inline

majority = sample_data.categories.value_counts()
majority_prop = majority /sum(majority)*100
best_cat = [majority_prop[majority_prop == max(majority_prop)].index.values.astype(str)[0]]

# Predict Training
pred_test_majo = best_cat * len(y_train)
print(f"\n Training results for Majority Baseline: \n {metrics.classification_report(y_train, pred_test_majo, digits=3)}"  )

# predict VALIDATION SET
pred_val_majo = best_cat * len(y_val)
print(metrics.classification_report(y_val, pred_val_majo, digits=3))

# TEST SET
pred_test_majo = best_cat * len(y_test)
print(f"\n Test results for Majority Baseline: \n {metrics.classification_report(y_test, pred_test_majo, digits=3)}"  )



#-----------------------------
# Preprocess to BOW and TFIDF
#-----------------------------
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# BOW
bow_vec = CountVectorizer(lowercase=True, 
                          strip_accents='ascii', 
                          stop_words='english', 
                          ngram_range=(1,2),
                          max_features=10000)
X_train_bow = bow_vec.fit_transform(X_train.abstract_tensor)
X_val_bow = bow_vec.fit_transform(X_val.abstract_tensor)
X_test_bow = bow_vec.fit_transform(X_test.abstract_tensor)

# See the matrix as a dataframe
X_train_bow_df = pd.DataFrame(X_train_bow.toarray(), columns=bow_vec.get_feature_names())


# TFIDF
tfidf_vec = TfidfVectorizer(lowercase=True, 
                            strip_accents='ascii', 
                            stop_words='english', 
                            ngram_range=(1,2),
                            max_features=10000)
X_train_tfidf = tfidf_vec.fit_transform(X_train.abstract_tensor)
X_val_tfidf = tfidf_vec.fit_transform(X_val.abstract_tensor)
X_test_tfidf = tfidf_vec.fit_transform(X_test.abstract_tensor)




#-----------------
# Naive bays model
#-----------------
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px

#------
# BOW
#------
# BOW Find bet alpha in validation
accuracy_bow = []
for i in np.linspace(0,1,11):
    clf_bow = MultinomialNB(alpha=i).fit(X_train_bow, y_train)
    predicted_bow = clf_bow.predict(X_val_bow)
    acc = accuracy_score(predicted_bow, y_val)
    accuracy_bow.append([i,acc])    
bow_alpha = pd.DataFrame(accuracy_bow, columns=['Alpha', 'Validation Accuracy'] )

fig = px.line(bow_alpha, 
              x="Alpha", 
              y="Validation Accuracy", 
              title='Validation Accuracy for Naive Bayes by Alpha in BOW')
fig.show()

# Train on best Alpha
best_a = bow_alpha[bow_alpha['Validation Accuracy'] == max(bow_alpha['Validation Accuracy'])]['Alpha'].values[0]
clf_bow = MultinomialNB(alpha=best_a).fit(X_train_bow, y_train)
predicted_bow = clf_bow.predict(X_val_bow)

# Get accuracy report training
predicted_bow_t = clf_bow.predict(X_train_bow)
print(f"\n Training results for BOW: \n {metrics.classification_report(y_train, predicted_bow_t , digits=3)}"  )
# Get accuracy report validation
print(f"\n Validation results for BOW: \n {metrics.classification_report(y_val, predicted_bow, digits=3)}"  )
cm = confusion_matrix(y_val, predicted_bow, labels=y_val.unique().tolist())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=y_val.unique().tolist())
disp.plot()


#--------
# TF_IDF
#--------
# TF_IDF Find bet alpha in validation
accuracy_tfidf = []
for i in np.linspace(0,1,11):
    clf_tfidf = MultinomialNB(alpha=i).fit(X_train_tfidf, y_train)
    predicted_tfidf = clf_tfidf.predict(X_val_tfidf)
    acc = accuracy_score(predicted_tfidf, y_val)
    accuracy_tfidf.append([i,acc])    
tfidf_alpha = pd.DataFrame(accuracy_tfidf, columns=['Alpha', 'Validation Accuracy'] )

fig = px.line(tfidf_alpha, 
              x="Alpha", 
              y="Validation Accuracy", 
              title='Validation Accuracy for Naive Bayes by Alpha in TF_IDF')
fig.show()

# Train on best Alpha
best_a = tfidf_alpha[tfidf_alpha['Validation Accuracy'] == max(tfidf_alpha['Validation Accuracy'])]['Alpha'].values[0]
clf_tfidf = MultinomialNB(alpha=best_a).fit(X_train_tfidf, y_train)
predicted_tfidf = clf_tfidf.predict(X_val_tfidf)

# Get accuracy report training
predicted_tfidf_t = clf_tfidf.predict(X_train_tfidf)
print(f"\n Training results for TF_IDF: \n {metrics.classification_report(y_train, predicted_tfidf_t, digits=3)}"  )
# Get accuracy report validation
print(f"\n Validation results for TF_IDF: \n {metrics.classification_report(y_val, predicted_tfidf, digits=3)}"  )



#---------------------
# RNN with Tensor Flow
#---------------------
import numpy as np
#import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.keras import layers
import timeit
print(device_lib.list_local_devices())

#tfds.disable_progress_bar()

X_pre_train, X_test, y_pre_train, y_test = train_test_split(sample_data[['id', 'title', 'abstract_tensor']], 
                                                    sample_data.cat_id, 
                                                    test_size=0.1, 
                                                    random_state=42)

# 20% for validation and 70% for training
X_train, X_val, y_train, y_val = train_test_split(X_pre_train, 
                                                    y_pre_train, 
                                                    test_size=0.2, 
                                                    random_state=42)


VOCAB_SIZE = 2000
embedding_dim = 128

# cant run TFIDF on GPU
with tf.device('/cpu:0'):
    encoder = layers.TextVectorization(max_tokens=VOCAB_SIZE,
                                        standardize ="lower_and_strip_punctuation",
                                        split="whitespace",
                                        output_mode = "tf_idf",
                                        ngrams=(1,2))
    encoder.adapt(np.array(X_train['abstract_tensor']))


# check what is inside the encoder
vocab = np.array(encoder.get_vocabulary())
vocab[:100]


# Number of batch size to determine
# Try with half of the sample aka 15k rows
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=embedding_dim,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)), #, return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation = 'softmax')
])
model.summary()



# predict on a sample text without padding.
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions[0])

# Our loss function
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save model artifac while training
checkpoint_path = "training_4cats_drop/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# Train the model with the new callback
# https://www.tensorflow.org/tutorials/keras/save_and_load

# 5000w = 242sec 10train 5 valid
# timer to see how long it takes to execute

max_r_train = 1000 
max_r_val = 200
num_epochs = 10
BATCH_SIZE = 32

# Run training on CPU
start = timeit.default_timer()

with tf.device('/cpu:0'):
    history = model.fit(np.array(X_train['abstract_tensor'].iloc[:max_r_train ]), 
                        np.array(y_train.iloc[:max_r_train ]), 
                        epochs=num_epochs,
                        batch_size=BATCH_SIZE, 
                        validation_data=(np.array(X_val['abstract_tensor'].iloc[:max_r_val]), 
                                         np.array(y_val.iloc[:max_r_val])), 
                        verbose=2,
                        callbacks=[cp_callback])

stop = timeit.default_timer()
print('Time: ', stop - start)

# Run training on GPU
start = timeit.default_timer()

with tf.device('/gpu:0'):
    history = model.fit(np.array(X_train['abstract_tensor'].iloc[:max_r_train ]), 
                        np.array(y_train.iloc[:max_r_train ]), 
                        epochs=num_epochs,
                        batch_size=BATCH_SIZE, 
                        validation_data=(np.array(X_val['abstract_tensor'].iloc[:max_r_val]), 
                                         np.array(y_val.iloc[:max_r_val])), 
                        verbose=2,
                        callbacks=[cp_callback])

stop = timeit.default_timer()
print('Time: ', stop - start)




# save results
hist_csv = pd.DataFrame(history.history)
hist_csv.to_csv("lstm[128],dense[64,4], 0.62 acc.csv", index=True)




import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis


class KNN_NLC_Classifer():
    def __init__(self, k=1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type

    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # returns the label with closest match.     
    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []

        for i in range(len(x_test)):
            bestgroup = []
            
            for j in range(self.x_train.shape[0]):
                temp = self.document_similarity(x_test[i], self.x_train[j])
                bestgroup.append([self.y_train[j], temp])
            
            # get best prediciton on k
            group_df = pd.DataFrame(bestgroup)
            group_df = group_df.sort_values([1], ascending=[False])
            pre_pred = group_df.iloc[:self.k][0].mode()
            
            # May get mode than one prediction so get only the best one.
            if len(pre_pred) >1:
                pred = group_df[0].iloc[:1].values[0]
            else:
                pred = pre_pred.values[0]
            y_predict.append(pred)
        return y_predict
        
    def convert_tag(self, tag):
            """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
            tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
            try:
                return tag_dict[tag[0]]
            except KeyError:
                return None
        
    def doc_to_synsets(self, doc):
        tokens = word_tokenize(doc+' ')
        
        l = []
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)
        
        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l 
    

    def similarity_score(self, s1, s2, distance_type = 'path'):
        """
        Calculate the normalized similarity score of s1 onto s2
        For each synset in s1, finds the synset in s2 with the largest similarity value.
        Sum of all of the largest similarity values and normalize this value by dividing it by the
        number of largest similarity values found.
    
        Args:
            s1, s2: list of synsets from doc_to_synsets
    
        Returns:
            normalized similarity score of s1 onto s2
        """
        s1_largest_scores = []
    
        for i, s1_synset in enumerate(s1, 0):
            max_score = 0
            for s2_synset in s2:
                if distance_type == 'path':
                    score = s1_synset.path_similarity(s2_synset, simulate_root = False)
                else:
                    score = s1_synset.wup_similarity(s2_synset)                  
                if score != None:
                    if score > max_score:
                        max_score = score
            
            if max_score != 0:
                s1_largest_scores.append(max_score)
        
        mean_score = np.mean(s1_largest_scores)
               
        return mean_score 

    def document_similarity(self,doc1, doc2):
          """Finds the symmetrical similarity between doc1 and doc2"""
    
          synsets1 = self.doc_to_synsets(doc1)
          synsets2 = self.doc_to_synsets(doc2)
          
          return (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2


#------------
# Find best K
#------------
from sklearn import metrics

max_train = 1000
max_val = 500

my_k = [1,3,5,7]
knn_res = []

for i in my_k:
    classifier = KNN_NLC_Classifer(k=i, distance_type='path')
    classifier.fit(np.array(X_train['abstract_tensor'].iloc[:max_train]), np.array(y_train.iloc[:max_train]))
    y_pred_final = classifier.predict(np.array(X_val['abstract_tensor'].iloc[:max_val]))
    score_knn = accuracy_score(y_val.iloc[:max_val], y_pred_final)
    print(score_knn)
    knn_res.append([i, score_knn])



#------------------------------------
# Fit KNN with our best K on TEST SET
#------------------------------------
start = timeit.default_timer()
classifier = KNN_NLC_Classifer(k=3, distance_type='path')
classifier.fit(np.array(X_train['abstract_tensor'].iloc[:max_train]), np.array(y_train.iloc[:max_train]))
y_pred_final = classifier.predict(np.array(X_test['abstract_tensor']))
print(f"\n Test Results for KNN: \n {metrics.classification_report(y_test, y_pred_final, digits=3)}"  )
stop = timeit.default_timer()
print('Time: ', stop - start)


#--------------------
# Test Set Evaluation
#--------------------
results_rnn = model.evaluate(np.array(X_test['abstract_tensor']),
                             np.array(y_test))
print("test loss, test acc:", results_rnn)


rnn_pred = model.predict(np.array(X_test['abstract_tensor']))
 
# save predictions
rnn_csv = pd.DataFrame(rnn_pred)
original_test = pd.concat([X_test, y_test], axis=1) 
original_test = pd.merge(original_test, unique_categories, on = 'cat_id', how = 'left')
original_test.reset_index(inplace=True)

original_test_res = pd.concat([original_test, rnn_csv], axis=1) 
original_test_res.to_csv("rnn_pred.csv", index=True)


# Generate RNN confusion matrix
rnn_pred = pd.read_csv('C:/Users/Vladi/OneDrive - HEC Montréal/ML 1 Group project/rnn_pred_match.csv')
rnn_pred.head()
cm = confusion_matrix(rnn_pred.true_l, rnn_pred.pred_l, labels=rnn_pred.true_l.unique().tolist())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rnn_pred.true_l.unique().tolist())
disp.plot()

print(f"\n Test Results for RNN: \n {metrics.classification_report(rnn_pred.true_l, rnn_pred.pred_l, digits=3)}"  )

# TFIDF confusion matrix
predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
print(f"\n Test Results for TFIDF: \n {metrics.classification_report(y_test, predicted_tfidf, digits=3)}"  )

# BOW confusion matrix
predicted_bow = clf_bow.predict(X_test_bow)
print(f"\n Test Results for BOW: \n {metrics.classification_report(y_test, predicted_bow, digits=3)}"  )


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
with tf.device('/cpu:0'):
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")
