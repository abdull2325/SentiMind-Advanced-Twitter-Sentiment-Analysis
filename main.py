import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess the text
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load and preprocess the data
def load_data(file_path, nrows=None):
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(file_path, encoding='latin-1', names=columns, nrows=nrows)
    df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})
    return df[['text', 'sentiment']]

# Function to preprocess a chunk of data
def preprocess_chunk(chunk):
    chunk['processed_text'] = chunk['text'].apply(preprocess_text)
    return chunk[['processed_text', 'sentiment']]

# Load the data
print("Loading and preprocessing data...")
file_path = '/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv'
df = load_data(file_path, nrows=1600000)
print("Data loaded.")

# Print value counts of sentiment
print("Sentiment distribution:")
print(df['sentiment'].value_counts())

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
plt.close()

# Ensure we have both positive and negative samples
if len(df['sentiment'].unique()) < 2:
    print("Error: Only one class present in the data. Attempting to balance the dataset.")
    df = load_data(file_path, nrows=None)  # Load all data
    print("New sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    if len(df['sentiment'].unique()) < 2:
        print("Error: Still only one class present. Check your data source.")
        exit()

# Balance the dataset if it's heavily skewed
min_class_count = df['sentiment'].value_counts().min()
df_balanced = df.groupby('sentiment').apply(lambda x: x.sample(min_class_count)).reset_index(drop=True)
print("Balanced sentiment distribution:")
print(df_balanced['sentiment'].value_counts())

# Preprocess the data using multiprocessing
print("Preprocessing data...")
num_processes = multiprocessing.cpu_count()
chunks = np.array_split(df_balanced, num_processes)

with multiprocessing.Pool(processes=num_processes) as pool:
    processed_chunks = pool.map(preprocess_chunk, chunks)

df_balanced = pd.concat(processed_chunks, ignore_index=True)
print("Data preprocessed.")

# Create word clouds
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

create_wordcloud(' '.join(df_balanced[df_balanced['sentiment'] == 'positive']['processed_text']), 'Positive Sentiment Word Cloud')
create_wordcloud(' '.join(df_balanced[df_balanced['sentiment'] == 'negative']['processed_text']), 'Negative Sentiment Word Cloud')

# Split the data
X = df_balanced['processed_text']
y = df_balanced['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize the text
print("Vectorizing the text...")
vectorizer = CountVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
print("Training the model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Visualize feature importance
feature_importance = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': model.coef_[0]
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Most Important Features')
plt.savefig('feature_importance.png')
plt.close()

# Function to predict sentiment for new text
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    return prediction, probabilities

# Example usage
sample_text = "I love this product! It's amazing and works perfectly."
sentiment, probabilities = predict_sentiment(sample_text)
print(f"Sample text: {sample_text}")
print(f"Predicted sentiment: {sentiment}")
for label, prob in zip(model.classes_, probabilities):
    print(f"{label}: {prob:.4f}")

# Visualize prediction probabilities using matplotlib
plt.figure(figsize=(10, 6))
plt.bar(model.classes_, probabilities)
plt.title('Prediction Probabilities')
plt.xlabel('Sentiment')
plt.ylabel('Probability')
plt.savefig('prediction_probabilities.png')
plt.close()

# Interactive sentiment analysis
def interactive_sentiment_analysis():
    while True:
        text = input("Enter a text to analyze (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        sentiment, probabilities = predict_sentiment(text)
        print(f"Predicted sentiment: {sentiment}")
        for label, prob in zip(model.classes_, probabilities):
            print(f"{label}: {prob:.4f}")
        print()

print("Starting interactive sentiment analysis. Type 'quit' to exit.")
interactive_sentiment_analysis()
