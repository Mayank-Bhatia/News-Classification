# news category classifier using SGD

#necesary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

news = pd.read_csv('data/uci-news-aggregator.csv') # import data
news['TITLE'] = news['TITLE'].str.replace('[^\w\s]','') # unpunctuate

vectorizer = CountVectorizer(stop_words='english') # setting stop-words, so words like "the" and "it" are ignored
X = vectorizer.fit_transform(news['TITLE']) # convert TITLE samples to vectors
y = news['CATEGORY'] # label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 30% split

# model (best params established through gridsearch in notebook)
sgd = SGDClassifier(n_jobs=-1, n_iter=10, alpha=1e-05, loss='hinge', random_state=1234)
sgd.fit(X_train, y_train)

# custom function that inputs a news title, and outputs one of 4 specified categories
def title_to_category(title):
    categories = {'b' : 'business', 
                  't' : 'science and technology', 
                  'e' : 'entertainment', 
                  'm' : 'health'}
    pridicter = sgd.predict(vectorizer.transform([title]))
    return categories[pridicter[0]]

# testing a headline from The Onion
print(title_to_category("Johnson & Johnson introduces 'nothing but tears shampoo' to 'toughen up' infants."))
# link to article: http://www.theonion.com/article/johnson-johnson-introduces-nothing-but-tears-shamp-2506

# output: health
