import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")
df = df[['Resume_str', 'Category']]  # Use correct column names
df.columns = ['resume_text', 'category']  # Rename for consistency

# Vectorize text
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['resume_text'])
y = df['category']

# Train classifier
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("✅ Model and TF-IDF saved successfully!")
