import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
import joblib

# Downloading stopwords
nltk.download('stopwords')

# Data Preprocessing
dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
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

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1420)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Saving BoW dictionary for later use
pickle.dump(cv, open('c1_BoW_Sentiment_Model.pkl', 'wb'))

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Exporting classifier
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')

# Model performance
y_pred = classifier.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
