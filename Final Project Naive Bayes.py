import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv('train.csv')

# Preprocess the data
data['email'] = data['email'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # Remove punctuation
data['email'] = data['email'].apply(lambda x: x.lower())  # Lowercase
data['email'] = data['email'].str.replace('[^\w\s]','')  # Remove unnecessary symbols

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Create a bag-of-words representation of the data
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data['email'])
test_vectors = vectorizer.transform(test_data['email'])

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_vectors, train_data['label'])

# Predict the labels of the test data
predictions = classifier.predict(test_vectors)

# Export the predictions to a CSV file
output_data = test_data.copy()
output_data['predicted_label'] = predictions
output_data.to_csv('email_classification_predictions_NB.csv', index=False)

# Print the accuracy of the classifier
accuracy = accuracy_score(test_data['label'], predictions)
print('Accuracy:', accuracy)