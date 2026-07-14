"""
NLP Topic Modeling for Customer Reviews

This project demonstrates an end-to-end NLP pipeline for
customer review analysis using BERTopic, BERT emotion analysis,
Large Language Models and Gensim LDA.

Dataset omitted due to confidentiality.
"""

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from IPython.display import display
from bertopic import BERTopic
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

Google_data = pd.read_excel("Google_12_months.xlsx")
Trustpilot_data = pd.read_excel("Trustpilot_12_months.xlsx")

print("Google Data\n")
print(f"view data shape : \n{Google_data.shape}")
print(f"view data frame : \n{Google_data.head()}")

print("Trustpilot Data\n")
print(f"view data shape : \n{Trustpilot_data.shape}")
print(f"view data frame : \n{Trustpilot_data.head()}")

Google_data.isna().sum()

Trustpilot_data.isna().sum()

# Remove rows where the Google review comment is missing.
cleaned_Google_data = Google_data.dropna(subset=['Comment'])

# Remove rows where the review text is missing, as these cannot be used for text analysis.
cleaned_Trustpilot_data = Trustpilot_data.dropna(subset=['Review Content'])

"""#Data Investigation"""

print("unique locations in Google data\n")
print("number of unique location : ",cleaned_Google_data['Club\'s Name'].nunique())
print("\n",cleaned_Google_data['Club\'s Name'].unique())

print("unique locations in Trustpilot data\n")
print("number of unique location : ",cleaned_Trustpilot_data['Location Name'].nunique())
print("\n",cleaned_Trustpilot_data['Location Name'].unique())

"""## Common Locations"""

# Identify and store the locations that are present in both the Google and Trustpilot datasets.
common_locations = set(cleaned_Google_data['Club\'s Name']) & set(cleaned_Trustpilot_data['Location Name'])
common_locations_df = pd.DataFrame(
    {'Location': sorted(common_locations)}
)
print(common_locations_df)

nltk.download('all')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()

# Load the English stopword list and initialise the lemmatiser.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert the review text to lowercase for consistency.
    text = str(text).lower()

    # Split the text into individual words (tokens).
    tokens = word_tokenize(text)

    # Remove stopwords, numbers, and non-alphabetic tokens.
    filtered_tokens = [
        token for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    # Convert words to their base form (e.g., "running" → "run").
    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    # Reconstruct the cleaned text from the processed tokens.
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

def print_top_topic_words(model, n_topics=2):
    """
    Print the top words for the top n BERTopic topics,
    excluding the outlier topic (-1).
    """

    topic_freq = model.get_topic_freq()
    top_topics = topic_freq[topic_freq["Topic"] != -1].head(n_topics)

    for topic in top_topics["Topic"]:
        words = [word for word, score in model.get_topic(topic)]

        print(f"Topic {topic}:")
        print(words)
        print()

from nltk import FreqDist

def get_word_frequency(processed_reviews):
    """
    Tokenize a collection of processed reviews and return
    the tokens and their frequency distribution.
    """
    words = word_tokenize(' '.join(processed_reviews))
    freq = FreqDist(words)
    return words, freq

def plot_top_words(freq_dist, title, top_n=10):
    # Get the top N words
    top_words = freq_dist.most_common(top_n)

    # Separate words and frequencies
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(words, counts)
    plt.title(title)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

from wordcloud import WordCloud

def plot_wordcloud(text_column, title):
    # Combine all reviews into one string
    text = ' '.join(text_column)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

cleaned_Google_data['Processed_Comment'] = cleaned_Google_data['Comment'].apply(preprocess_text)

cleaned_Trustpilot_data['Processed_Review'] = cleaned_Trustpilot_data['Review Content'].apply(preprocess_text)

google_words, google_freq = get_word_frequency(cleaned_Google_data['Processed_Comment'])
trustpilot_words, trustpilot_freq = get_word_frequency(cleaned_Trustpilot_data['Processed_Review'])

print("Top 10 words in Google data:")
print(google_freq.most_common(10))

print("\nTop 10 words in Trustpilot data:")
print(trustpilot_freq.most_common(10))

plot_top_words(google_freq, 'Top 10 Most Frequent Words - Google Comments')
plot_top_words(trustpilot_freq, 'Top 10 Most Frequent Words - Trustpilot Reviews')

plot_wordcloud(cleaned_Google_data['Processed_Comment'],'Word Cloud - Google Reviews')
plot_wordcloud(cleaned_Trustpilot_data['Processed_Review'],'Word Cloud - Trustpilot Reviews')

"""Both datasets show a very similar word frequency distribution. The word gym is the most frequent by a large margin, followed by words such as equipment, staff, class, clean, and friendly. This suggests that customers on both Google and Trustpilot discuss similar aspects of their gym experience, particularly facilities, equipment, staff, and cleanliness.

## Negative Comments & Reviews
"""

negative_Google_data = cleaned_Google_data[cleaned_Google_data['Overall Score'] < 3]
negative_Trustpilot_data = cleaned_Trustpilot_data[cleaned_Trustpilot_data['Review Stars'] < 3]

negative_google_words, negative_google_freq = get_word_frequency(negative_Google_data['Processed_Comment'])
negative_trustpilot_words, negative_trustpilot_freq = get_word_frequency(negative_Trustpilot_data['Processed_Review'])

plot_top_words(negative_google_freq,'Top 10 Words in Negative Google Reviews')
plot_top_words(negative_trustpilot_freq,'Top 10 Words in Negative Trustpilot Reviews')

plot_wordcloud(negative_Google_data['Processed_Comment'],'Word Cloud - Negative Google Reviews')
plot_wordcloud(negative_Trustpilot_data['Processed_Review'],'Word Cloud - Negative Trustpilot Reviews')

"""The negative review word distributions are similar across both datasets, with gym being the most frequent word. However, Google negative reviews place greater emphasis on staff, people, and equipment, suggesting that customers often complain about interactions with staff and the availability or condition of equipment. In contrast, Trustpilot reviews feature membership more prominently, indicating that account and membership-related issues are discussed more frequently on that platform. Overall, both datasets highlight common concerns such as equipment, waiting times, and gym facilities, while each platform emphasises slightly different aspects of the customer experience.

#Topic Modelling

##Negative reviews on common Locations
"""

negative_common_google = negative_Google_data[
    negative_Google_data["Club's Name"].isin(common_locations)
]

negative_common_trustpilot = negative_Trustpilot_data[
    negative_Trustpilot_data["Location Name"].isin(common_locations)
]

processed_google_reviews = negative_common_google["Processed_Comment"].tolist()
processed_trustpilot_reviews = negative_common_trustpilot["Processed_Review"].tolist()

processed_merged_reviews = processed_google_reviews + processed_trustpilot_reviews
processed_merged_reviews

negative_common_reviews_model = BERTopic(verbose=True)
negative_common_reviews_model.fit(processed_merged_reviews)
negative_common_topics, negative_common_probabilities = negative_common_reviews_model.transform(processed_merged_reviews)

negative_common_topic_freq = negative_common_reviews_model.get_topic_freq().head(10)
negative_common_topic_freq

"""The majority of reviews were assigned to Topic -1, indicating that many reviews did not fit clearly into a single topic. The remaining reviews were grouped into several topics with document frequencies ranging from 71 to 203, suggesting that multiple recurring themes are present in the dataset."""

print_top_topic_words(negative_common_reviews_model, n_topics=2)

negative_common_reviews_model.visualize_topics()

"""The intertopic distance map shows that BERTopic identified several distinct topic clusters. Topics that are close together represent related themes, while topics that are further apart represent different aspects of customer feedback. The larger circles correspond to topics containing more reviews. Overall, the map indicates that the reviews cover multiple recurring themes, with some topics being more prominent than others."""

negative_common_reviews_model.visualize_barchart()

negative_common_reviews_model.visualize_heatmap()

negative_common_reviews_model.get_topic_info().head(10)

"""BERTopic identified several distinct themes within the negative reviews from the common locations. The main topics relate to air conditioning and ventilation, fitness classes and bookings, gym access (codes and PINs), parking, cleanliness of toilets and changing facilities, staff behaviour and customer service, noise levels, gym opening hours, and broken equipment. These topics indicate that customer dissatisfaction is primarily driven by operational and facility-related issues rather than a single recurring problem. In addition, Topic -1 represents outlier reviews that did not fit clearly into any of the identified themes.

#Further Data Investigation

## Top 20 locations with the highest number of negative reviews
"""

google_top20 = (
    negative_Google_data["Club's Name"]
    .value_counts()
    .head(20)
)

print(google_top20)

trustpilot_top20 = (
    negative_Trustpilot_data["Location Name"]
    .value_counts()
    .head(20)
)

print(trustpilot_top20)

common_locations = set(google_top20.index) & set(trustpilot_top20.index)

print(common_locations)

"""The top 20 locations with the highest number of negative reviews show some overlap between Google and Trustpilot. Five locations appear in the top 20 lists of both datasets, while the remaining locations differ. This suggests that there is some agreement between the two platforms regarding locations that receive higher levels of negative feedback, but there are also notable differences. Additionally, many of the locations in both top 20 lists are different branches located in the London area. Although these are not always the same clubs, this indicates that negative reviews are concentrated across several London locations on both platforms. The remaining differences may be due to variations in the number of reviews, the user base of each platform, or differences in customer reviewing behaviour."""

# Count the number of Google/Trustpilot reviews for each gym location and store the results in a DataFrame
google_counts = (
    cleaned_Google_data["Club's Name"]
    .value_counts()
    .rename_axis("Location")
    .reset_index(name="Google Comments")
)

trustpilot_counts = (
    cleaned_Trustpilot_data["Location Name"]
    .value_counts()
    .rename_axis("Location")
    .reset_index(name="Trustpilot Reviews")
)

print(google_counts.head())
print(google_counts.columns)

print(trustpilot_counts.head())
print(trustpilot_counts.columns)

#merge the 2 data sets based on location
merged_counts = pd.merge(
    google_counts,
    trustpilot_counts,
    on="Location",
    how="outer"
)
merged_counts

merged_counts.isna().sum()

# Filling NA values with 0
merged_counts = merged_counts.fillna(0)

merged_counts.head()

# Convert the review count columns to integers, calculate the total number of
# reviews for each location, and sort the locations by total review count.

merged_counts[["Google Comments", "Trustpilot Reviews"]] = (
    merged_counts[["Google Comments", "Trustpilot Reviews"]].astype(int)
)

merged_counts["Total Reviews"] = (
    merged_counts["Google Comments"] + merged_counts["Trustpilot Reviews"]
)

merged_counts = merged_counts.sort_values(
    by="Total Reviews",
    ascending=False
)

print(merged_counts.head(30))

top30_locations = merged_counts.head(30)["Location"].tolist()

# Filter both datasets to retain only reviews from the top 30 locations
# with the highest total number of reviews.

google_top30 = cleaned_Google_data[
    cleaned_Google_data["Club's Name"].isin(top30_locations)
]

trustpilot_top30 = cleaned_Trustpilot_data[
    cleaned_Trustpilot_data["Location Name"].isin(top30_locations)
]

top30_google_words, top30_google_freq = get_word_frequency(google_top30['Processed_Comment'])
top30_trustpilot_words, top30_trustpilot_freq = get_word_frequency(trustpilot_top30['Processed_Review'])

plot_top_words(top30_google_freq,"Top 10 Words - Top 30 Google Locations")
plot_wordcloud(google_top30["Processed_Comment"],"Word Cloud - Top 30 Google Locations")

plot_top_words(top30_trustpilot_freq,"Top 10 Words - Top 30 Trustpilot Locations")
plot_wordcloud(trustpilot_top30["Processed_Review"],"Word Cloud - Top 30 Trustpilot Locations")

"""The results are broadly similar to the first run, with gym, class, people, machine, and equipment remaining the most frequent words. However, the analysis of the top 30 locations places greater emphasis on words such as time, clean, and need, indicating slight differences in customer discussions within the most-reviewed locations. Overall, the dominant themes remain consistent across both analyses."""

top30_combined_reviews = (
    google_top30["Processed_Comment"].tolist() +
    trustpilot_top30["Processed_Review"].tolist()
)

top30_model = BERTopic(verbose=True)
top30_model.fit(top30_combined_reviews)
top30_topics, top30_probabilities = top30_model.fit_transform(top30_combined_reviews)

top30_combined_topic_freq = top30_model.get_topic_freq().head(10)
top30_combined_topic_freq

print_top_topic_words(top30_model, n_topics=2)

"""The dominant topics focus on personal training and fitness classes. Unlike the earlier analysis of negative reviews, the top 30 combined reviews highlight more positive customer experiences, with customers frequently praising personal trainers, instructors, and fitness classes. The appearance of instructor names within the topics suggests that individual staff members make a significant contribution to customer satisfaction at the most-reviewed locations."""

top30_model.visualize_topics()

top30_model.visualize_barchart()

top30_model.visualize_heatmap()

"""These visualisations support the topic analysis, showing that positive themes related to personal training, fitness classes, friendly staff, and cleanliness form coherent but distinct clusters. Other topics, such as customer service and overcrowding during peak hours, remain separate, indicating that BERTopic successfully distinguishes between different aspects of the customer experience."""

top30_model.get_topic_info().head(10)

negative_common_reviews_model.get_topic_info().head(10)

"""The second BERTopic analysis produced different results from the first run because it included all reviews from the top 30 locations rather than only negative reviews. The dominant topics focused on personal training, fitness classes, friendly staff, and clean facilities, reflecting generally positive customer experiences. Several topics also highlighted customer service and overcrowding during peak hours, indicating that while members frequently praise the quality of the facilities and staff, operational challenges remain. Overall, this analysis provides a more balanced view of customer experiences by capturing both the strengths of the most-reviewed locations and the recurring issues identified in the earlier negative review analysis.

#Emotion Analysis
"""

from transformers import pipeline

def get_top_emotion(text):
    # Predict the emotions for the review.
    emotion_labels = emotion_classifier(
        str(text),
        truncation=True,
        max_length=512
    )[0]

    # Return the emotion with the highest confidence score.
    return max(emotion_labels, key=lambda x: x["score"])["label"]

def plot_emotions(emotion_counts, title):
    """
        Plot the distribution of emotions as a bar chart.
    """
    emotion_counts.plot.bar(figsize=(7,4))
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Number of Reviews")
    plt.show()

emotion_classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', top_k=None)

sentence = "I am extremely disappointed with the customer service."
emotion_labels = emotion_classifier(sentence)
emotion_labels_sorted = sorted(emotion_labels[0], key=lambda x: x["score"], reverse=True)

print(emotion_labels_sorted)

google_negative_sample = negative_Google_data.sample(
    n=1000,
    random_state=42
)

trustpilot_negative_sample = negative_Trustpilot_data.sample(
    n=1000,
    random_state=42
)

google_negative_sample["Top Emotion"] = google_negative_sample["Comment"].apply(get_top_emotion)
google_negative_sample["Top Emotion"]

trustpilot_negative_sample["Top Emotion"] = trustpilot_negative_sample["Review Content"].apply(get_top_emotion)
trustpilot_negative_sample["Top Emotion"]

google_emotions = google_negative_sample["Top Emotion"].value_counts()
trustpilot_emotions = trustpilot_negative_sample["Top Emotion"].value_counts()

plot_emotions(google_emotions,"Google Negative Reviews Emotion Distribution")

plot_emotions(trustpilot_emotions,"Trustpilot Negative Reviews Emotion Distribution")

"""Both datasets show a similar emotion distribution, with anger being the most common emotion in negative reviews. Google reviews contain slightly more sadness than joy, whereas Trustpilot reviews show slightly more joy than sadness. This may be because some customers acknowledge positive aspects of their experience before describing their complaints. Overall, the emotion patterns are consistent across both platforms, with fear, love, and surprise appearing relatively infrequently."""

google_anger = google_negative_sample[google_negative_sample["Top Emotion"] == "anger"]
google_anger

trustpilot_anger = trustpilot_negative_sample[trustpilot_negative_sample["Top Emotion"] == "anger"]
trustpilot_anger

anger_reviews = (
    google_anger["Processed_Comment"].tolist()
    +
    trustpilot_anger["Processed_Review"].tolist()
)

anger_model = BERTopic(verbose=True)
anger_topics, anger_probabilities = anger_model.fit_transform(anger_reviews)

print_top_topic_words(anger_model, n_topics=2)

"""Topic 0 represents complaints related to membership administration, including joining, fees, account management, cancellations, refunds, and email communication. Topic 1 primarily contains non-English words, suggesting that multilingual reviews were grouped into a separate topic rather than representing a specific customer issue."""

anger_model_words, anger_model_freq = get_word_frequency(anger_reviews)

anger_model.get_topic_freq().head(10)

plot_top_words(anger_model_freq,"Top 10 Words in Anger Reviews")
plot_wordcloud(anger_reviews,"Word Cloud - Anger Reviews")

anger_model.visualize_topics()

anger_model.visualize_barchart()

"""The anger-focused BERTopic model identified more specific complaint themes than the previous analyses. The dominant topic focuses on membership administration, including joining, account management, fees, cancellations, refunds, and email communication. These findings indicate that administrative processes are a key driver of customer anger, while a second topic reflects the multilingual nature of the review data rather than a specific customer concern."""

anger_model.visualize_heatmap()

"""The BERTopic results for the anger-only reviews are more focused than those from the top 30 locations. While the top 30 analysis highlights a mixture of positive themes, such as personal training, fitness classes, friendly staff, and clean facilities, the anger-only analysis is dominated by membership administration issues. The main complaints relate to joining, membership fees, account management, cancellations, refunds, and email communication. A second topic consists primarily of multilingual reviews rather than a specific customer concern. These findings demonstrate that filtering reviews by emotion helps identify the issues most likely to trigger customer frustration and provides a clearer understanding of the primary causes of angry reviews.

#Using a large language model
"""

import os
os.environ.update({'LD_LIBRARY_PATH': '/usr/lib64-nvidia'})

import ollama

from ollama import Client
client = Client(host="http://localhost:11434", timeout=120)

def run_llm_chat(messages: list,
                     model: str = "phi4-mini",
                     temperature: float = 0.2) -> str:

    """
    Use ollama.chat;
    """
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature, "num_ctx": 2192, "num_predict": 120}, # Increased num_predict further
    )

    return response.message.content

def run_prompt(prompt, model="phi4-mini"):
    messages = [
        {"role": "user", "content": prompt}
    ]

    return run_llm_chat(messages, model=model)

def extract_topics(review):
    prompt = f"""
      In the following customer review, pick out the main 3 topics.
      Return them in a numbered list format, with each one on a new line.

      Review:
      {review}
      """

    return run_prompt(prompt)

def generate_recommendations(topics_text):
    prompt = f"""
    For the following text topics obtained from negative customer reviews, give 5 concise actionable insights for this gym company.

    Topics:
    {topics_text}

    Return only 5 numbered suggestions. Keep each suggestion under 20 words.
    """

    return run_prompt(prompt)

# Combine angry reviews from both Google and Trustpilot into a single list,
# then select the first 20 reviews as a sample for topic extraction using Phi-4-mini.

bad_reviews_sample = (
    google_anger["Comment"].tolist() +
    trustpilot_anger["Review Content"].tolist()
)[:20]

phi_results = []

for review in bad_reviews_sample:
    phi_results.append(extract_topics(review))

# Extract the topics generated for each review and combine them into a single
# list that will be used as input for the subsequent BERTopic analysis.

comprehensive_topics = []

for result in phi_results:
    lines = result.split("\n")

    for line in lines:
        line = line.strip()

        # Keep only numbered topics
        if line.startswith(("1.", "2.", "3.")):
            comprehensive_topics.append(line)

print(comprehensive_topics[:20])

clean_topics = []

for topic in comprehensive_topics:
    clean_topics.append(topic.split(".", 1)[1].strip())

phi_extracted_topics_model = BERTopic(verbose=True)

phi_extracted_topics, phi_extracted_probabilities = (phi_extracted_topics_model.fit_transform(clean_topics))

phi_extracted_topics_model.get_topic_info().head(10)

print_top_topic_words(phi_extracted_topics_model, n_topics=2)

"""The BERTopic model built on the LLM-extracted topics identified two main themes. Topic 0 focuses on operational issues, particularly equipment quality and shower facilities, while Topic 1 captures broader customer satisfaction and service-related concerns. Compared with the BERTopic analysis performed on the original reviews, the LLM-generated topics are more concise but also more general, reflecting the summarised nature of the extracted topics."""

# Remove numbering from the extracted topics, eliminate duplicate topics,
# and combine the unique topics into a single text prompt for the LLM.
short_topics = []

for topic in comprehensive_topics:
    topic = topic.split(".", 1)[-1].strip()
    short_topics.append(topic)

short_topics = list(dict.fromkeys(short_topics))

#first 30 unique topics and combines them into one string separated by new lines
topics_text = "\n".join(short_topics[:30])

phi_recommendations = generate_recommendations(topics_text)
print(phi_recommendations)

"""Phi-4-mini generated five actionable recommendations based on the extracted customer review topics. The suggestions focus on improving shower facilities, enhancing customer service, investing in equipment maintenance, managing overcrowding, and communicating gym policies more effectively. These recommendations closely align with the issues identified through BERTopic and emotion analysis, demonstrating that the LLM can translate customer feedback into practical business actions.

#Gensim (LDA)
"""

from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Combine processed negative reviews from both datasets
negative_combined_reviews = (
    negative_Google_data["Processed_Comment"].tolist()
    +
    negative_Trustpilot_data["Processed_Review"].tolist()
)

# Tokenise each review into a list of words
negative_tokenised_reviews = [
    review.split()
    for review in negative_combined_reviews
]

# Create a dictionary mapping each word to a unique ID
dictionary = corpora.Dictionary(negative_tokenised_reviews)

# Optional: remove very rare and very common words
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Convert tokenised reviews into bag-of-words format
corpus = [
    dictionary.doc2bow(review)
    for review in negative_tokenised_reviews
]

#Run LDA with 10 topics

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    random_state=42,
    passes=10
)

# print topics
for idx, topic in lda_model.print_topics(num_topics=10, num_words=10):
    print(f"Topic {idx}: {topic}")

"""The LDA model identified several meaningful themes within the negative reviews, including fitness classes, shower and changing facilities, customer service, membership management, staff behaviour, equipment maintenance, and parking. These themes are broadly consistent with those identified by BERTopic, suggesting that both topic modelling approaches capture similar operational issues. Two topics consisted primarily of non-English words, reflecting the multilingual nature of the review data rather than specific customer concerns."""

#Visualise with pyLDAvis
pyLDAvis.enable_notebook()

lda_vis = gensimvis.prepare(
    lda_model,
    corpus,
    dictionary
)

lda_vis

"""Overall, the LDA visualisation identifies many of the same themes as BERTopic, including memberships, equipment, facilities, classes, and customer service. However, several topics overlap considerably, indicating that LDA separates themes less clearly than BERTopic. BERTopic produced more coherent and interpretable clusters, while LDA provided a useful confirmation of the main discussion themes."""