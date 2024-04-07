import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# loading the dataset
df = pd.read_csv('C:\\Users\\R.ARJUN\\OneDrive\\Desktop\\arjunp\\Machine Learning Internships\\Spam mail detection\\spam.csv',encoding = 'latin-1')
#if we load it directly it getting encoding because i used encoding attribute, this suggestion got from colab 

# Summarizing the Dataset
print(df.head(5)) # Printing Top 5 rows
print('\n\n\n',df.info()) # Printing Dataset information
 
# text preprocessing
df['text'] = df['text'].str.lower()  # it Converts all text's into lowercase

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# training the Naive_bayes classifier algorithm
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Make prediction
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
