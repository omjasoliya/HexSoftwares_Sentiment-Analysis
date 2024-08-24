import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import joblib

# Data Preprocessing
dataset = pd.read_csv('a2_RestaurantReviews_FreshDump.tsv', delimiter='\t', quoting=3)
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Loading the saved BoW model and classifier
cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', 'rb'))
classifier = joblib.load('c2_Classifier_Sentiment_Model')

# Transforming the new data and making predictions
X_fresh = cv.transform(corpus).toarray()
y_pred = classifier.predict(X_fresh)

# Adding predictions to the dataset
dataset['predicted_label'] = y_pred
dataset.to_csv('c3_Predicted_Sentiments_Fresh_Dump.tsv', sep='\t', encoding='UTF-8', index=False)

print("Predictions saved to 'c3_Predicted_Sentiments_Fresh_Dump.tsv'")
