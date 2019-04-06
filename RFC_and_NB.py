import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


df = pd.read_csv("D:/Semester 2/Natural Language Processing/trainSplit.csv", sep = ",")
df["comment_text"].head()
#print(df)

df["comment_text"]= df["comment_text"].str.lower()

#df["comment_text"] = re.sub(r"\d+","", df["comment_text"])
df["comment_text"] = df["comment_text"].str.replace(r"\d+","")
#df["comment_text"].head()
#print(df)

df["comment_text"] = df["comment_text"].str.replace(r"[^a-z]+"," ")
#df["comment_text"].head()

df["comment_text"] = df["comment_text"].str.strip()
df["comment_text"].head()

stopset = set(stopwords.words('english'))
df["commentText_noStopwords"] = df["comment_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopset)]))
#df["commentText_noStopwords"].head()

#df["tokenized_comment_text"] = df.apply(lambda row: nltk.word_tokenize(df["commentText_noStopwords"]), axis=1)
df["tokenized_comment_text"] = df.apply(lambda row: nltk.word_tokenize(row["commentText_noStopwords"]), axis=1)
#df["tokenized_comment_text"].head()

stemmer = PorterStemmer()
df["stemmed"] = df["tokenized_comment_text"].apply(lambda x: [stemmer.stem(y) for y in x])
df['stemmed'] = df['stemmed'].apply(' '.join)

# =============================================================================
#print(df['stemmed'].apply(' '.join))
# =============================================================================

corpus = (df['stemmed'])

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(corpus)
#print(vectorizer.vocabulary_)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts).toarray()

y_toxic = list(df['toxic'])

X_train, X_test, ytoxic_train, ytoxic_test = train_test_split(X_tfidf, y_toxic, train_size=0.75, test_size=0.25, random_state=1)

model_NB = MultinomialNB(alpha = 1.0, class_prior=None, fit_prior=False).fit(X_train,ytoxic_train)
model_RF = RandomForestClassifier(criterion= 'gini', max_depth= 75, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 20).fit(X_train,ytoxic_train)

predict_NB = model_NB.predict(X_test)
predict_RF = model_RF.predict(X_test)
print(np.mean(predict_NB == ytoxic_test))
print(np.mean(predict_RF == ytoxic_test))

print(confusion_matrix(ytoxic_test, predict_NB))
print(confusion_matrix(ytoxic_test, predict_RF))






