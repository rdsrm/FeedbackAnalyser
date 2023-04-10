# Import packages
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download NLTK packages
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

# Read data from SQL database
db_file = "feedback_data.sqlite"
conn = sqlite3.connect(db_file)
data = pd.read_sql_query("SELECT * FROM feedback", conn)
conn.close()

# Tokenize words and remove stopwords
stop_words = set(stopwords.words("english"))
all_words = [word.lower() for feedback in feedback_data for word in word_tokenize(feedback) if word.isalpha()]
filtered_words = [word for word in all_words if word not in stop_words and word not in ['system', 'software']]

# Calculate word frequency
word_count = Counter(filtered_words)

# Create a WordCloud
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words, min_font_size=10).generate_from_frequencies(word_count)

# Display the WordCloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(feedback) for feedback in feedback_data]
sentiment_df = pd.DataFrame(sentiment_scores)

# Display average sentiment scores
average_sentiment = sentiment_df.select_dtypes(include=np.number).mean()
print("Average Sentiment Scores:")
display(average_sentiment)

# Categorize feedback into positive, negative, and neutral
sentiment_categories = []
for score in sentiment_df['compound']:
    if score > 0.05:
        sentiment_categories.append('Positive')
    elif score < -0.05:
        sentiment_categories.append('Negative')
    else:
        sentiment_categories.append('Neutral')

# Count the number of feedback in each sentiment category
sentiment_counts = Counter(sentiment_categories)

# Create a pie chart of sentiment categories
labels = ['Positive', 'Negative', 'Neutral']
sizes = [sentiment_counts[label] for label in labels]
colors = ['#99ff99', '#ff9999', '#ffff99']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Sentiment Distribution", fontweight='bold')
plt.show()

# Calculate sentiment scores for each word
word_sentiment_scores = {word: analyzer.polarity_scores(word)['compound'] for word in word_count}


# Filter positive and negative words
positive_words = {word: score for word, score in word_sentiment_scores.items() if score > 0.05}
negative_words = {word: score for word, score in word_sentiment_scores.items() if score < -0.05}

# Sort the dictionaries
sorted_positive_words = sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:20]
sorted_negative_words = sorted(negative_words.items(), key=lambda x: x[1])[:20]

# Create DataFrames
top_positive_words_df = pd.DataFrame(sorted_positive_words, columns=['Word', 'Sentiment Score'])
top_positive_words_df['Count'] = top_positive_words_df['Word'].apply(lambda x: word_count[x])

top_negative_words_df = pd.DataFrame(sorted_negative_words, columns=['Word', 'Sentiment Score'])
top_negative_words_df['Count'] = top_negative_words_df['Word'].apply(lambda x: word_count[x])


# Display the top 20 positive words with bold header
(top_positive_words_df
 .style
 .set_table_styles([{
     'selector': 'th',
     'props': [('font-weight', 'bold')]
 }])
 .set_caption('Top 20 Positive Words'))

 # Display the top 20 negative words with bold header
(top_negative_words_df
 .style
 .set_table_styles([{
     'selector': 'th',
     'props': [('font-weight', 'bold')]
 }])
 .set_caption('Top 20 Negative Words'))

 # Find feedback with the word 'problematic'
problematic_comments = [feedback for feedback in feedback_data if 'problematic' in feedback.lower()]


# Display the top 20 problematic feedback
problematic_comments_df = pd.DataFrame(problematic_comments, columns=['Feedback']).head(20)
pd.set_option('display.max_colwidth', None)  # Remove the limit on column width

print("Top 20 Feedback Comments with the Word 'Problematic':")
display(problematic_comments_df)

# Find feedback with the word 'great'
great_comments = [feedback for feedback in feedback_data if 'great' in feedback.lower()]


# Display the top 20 great feedback
great_comments_df = pd.DataFrame(great_comments, columns=['Feedback']).head(20)
pd.set_option('display.max_colwidth', None)  # Remove the limit on column width

# Display the top 20 great feedback with a bold header
(great_comments_df
 .style
 .set_table_styles([{
     'selector': 'th',
     'props': [('font-weight', 'bold')]
 }])
 .set_caption("<b>Top 20 Feedback Comments with the Word 'Great'<b>"))


