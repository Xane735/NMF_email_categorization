import pandas as pd
import re
import string

# Load the dataset
df = pd.read_csv('enron_spam_data.csv')

# Display the first few rows
print(df.head())

# Define a text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Combine Subject and Message columns, handle NaNs
df['combined_text'] = (df['Subject'].fillna('') + ' ' + df['Message'].fillna(''))

# Clean the combined text
df['cleaned_text'] = df['combined_text'].apply(clean_text)
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit and transform the cleaned text
tfidf = vectorizer.fit_transform(df['cleaned_text'])
from sklearn.decomposition import NMF

# Define the number of topics
n_topics = 5

# Initialize and fit the NMF model
nmf_model = NMF(n_components=n_topics, random_state=42)
nmf_model.fit(tfidf)
# Get the vocabulary of terms
feature_names = vectorizer.get_feature_names_out()

# Display top keywords for each topic
for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"Topic #{topic_idx + 1}:")
    print(", ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))
    print()
"""import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Generate word clouds for each topic
for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"Topic #{topic_idx + 1} Word Cloud:")
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies({feature_names[i]: topic[i] for i in topic.argsort()[:-11:-1]})
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic #{topic_idx + 1}")
    plt.show()
"""
"""import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Example: If you want to count these specific words
target_words = ['gas', 'allocation', 'nomination', 'issue', 'production', 'tree', 'farm']

# Flatten all cleaned text into one list of words
all_words = ' '.join(df['cleaned_text'].dropna()).split()

# Count all word occurrences
word_counts = Counter(all_words)

# Filter only the target words
filtered_counts = {word: word_counts[word] for word in target_words}

# Sort the dictionary by frequency
filtered_counts = dict(sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True))

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(filtered_counts.keys(), filtered_counts.values(), color='teal')
plt.title('Frequency of Selected Words in Emails')
plt.xlabel('Words')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

n_top_words = 10
feature_names = vectorizer.get_feature_names_out()
all_words = ' '.join(df['cleaned_text'].dropna()).split()
word_counts = Counter(all_words)

num_topics = nmf_model.components_.shape[0]

# Set up the figure and axes — adjust rows & cols as needed
cols = 2
rows = (num_topics + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()  # Flatten in case of multiple rows

for topic_idx, topic in enumerate(nmf_model.components_):
    top_indices = topic.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_indices]
    counts = [word_counts[word] for word in top_words]

    ax = axes[topic_idx]
    ax.bar(top_words, counts, color='mediumseagreen')
    ax.set_title(f"Topic #{topic_idx + 1} - Keyword Frequencies")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

# If there are any unused subplots, remove them
for i in range(num_topics, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
