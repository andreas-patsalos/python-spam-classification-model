import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Loading the Training CSV File
train_data = pd.read_csv('train.csv')

# Removing unlabaled samples
train_data = train_data[train_data['label'].isin(['spam', 'ham'])]

# Preprocessing (Removing Punctuation, Lowercase and Removing unnecessary symbols)
train_data['email'] = train_data['email'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))  
train_data['email'] = train_data['email'].apply(lambda x: x.lower()) 
train_data['email'] = train_data['email'].str.replace('[^\w\s]','')

# Get a sample of 10 emails
sample = train_data['email'].sample(n=10)
print("Sample of 10 emails:")
print(sample.to_string(index=False))

# Transforming the email column attribute into plain words
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data['email'])

# Feeding the training data to a Multinomial Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(train_vectors, train_data['label'])

# Insert testing CSV file
test_data = pd.read_csv('train.csv')

# Preprocess the unabaled CSV file
test_data['email'] = test_data['email'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
test_data['email'] = test_data['email'].apply(lambda x: x.lower())
test_data['email'] = test_data['email'].str.replace('[^\w\s]','')

# Transforming the email column attribute into plain words
test_vectors = vectorizer.transform(test_data['email'])

# Predicting the labels
predictions = classifier.predict(test_vectors)

# Calculation and printing of the accuracy
accuracy = accuracy_score(test_data['label'], predictions)
print(f"Accuracy: {accuracy:.2f}")

# Exporting the predictions to a CSV file
output_data = test_data.copy()
output_data['predicted_label'] = predictions
output_data.to_csv('email_classification_predictions_FINAL.csv', index=False)
