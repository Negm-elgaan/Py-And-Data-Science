# Import nltk
import nltk
# Download the punkt_tab package 
nltk.download("punkt_tab")

text = """
The stock market saw a significant dip today. Experts believe the downturn may continue.
However, many investors are optimistic about future growth.
"""

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)
print(sentences)
######################
feedback = "I reached out to support and got a helpful response within minutes!!! Very #impressed"

# Tokenize the text
tokens = word_tokenize(feedback)

# Get the list of English stop words
stop_words = stopwords.words('english')

# Remove stop words 
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)
#############################
import string

filtered_tokens = ['reached', 'support', 'got', 'helpful', 'response', 'within', 'minutes', '!', '!', '!', '#', 'impressed']

# Remove punctuation
clean_tokens = [word for word in filtered_tokens if word not in string.punctuation]

print(clean_tokens)
#########################
review = "I have been FLYING a lot lately and the Flights just keep getting DELAYED. Honestly, traveling for WORK gets exhausting with endless delays, but every trip teaches you something new!"

# Lowercase the review
lower_text = review.lower()

# Tokenize the lower_text into words
tokens = word_tokenize(lower_text)

# Remove stop words and punctuation
clean_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

print(clean_tokens)
###############################
from nltk.stem import PorterStemmer
clean_tokens = ['flying', 'lot', 'lately', 'flights', 'keep', 'getting', 'delayed', 'honestly', 'traveling', 'work', 'gets', 'exhausting', 'endless', 'delays', 'every', 'travel', 'teaches', 'something', 'new']

# Create stemmer
stemmer = PorterStemmer()

# Stem each token
stemmed_tokens = [stemmer.stem(word) for word in clean_tokens]

print(stemmed_tokens)
######################################
from nltk.stem import WordNetLemmatizer

clean_tokens = ['flying', 'lot', 'lately', 'flights', 'keep', 'getting', 'delayed', 'honestly', 'traveling', 'work', 'gets', 'exhausting', 'endless', 'delays', 'every', 'travel', 'teaches', 'something', 'new']

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize each token
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]

print(lemmatized_tokens)
###############################
reviews = [
    "The product is fantastic! It works like a charm.",
    "I hated the product. It broke after one use.",
    "Product was okay, not the best, but fine overall."
]
# Preprocess the reviews
cleaned_reviews = [preprocess(review) for review in reviews]

vectorizer = CountVectorizer()
# Fit the vectorizer
vectorizer.fit(cleaned_reviews)
# Print the vocabulary 
print(vectorizer.get_feature_names_out())
#############################################
# Transform the reviews
bow_matrix = vectorizer.transform(cleaned_reviews)

# Print the BoW representation
print(bow_matrix.toarray())
########################################################
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    return " ".join(tokens)
  
cleaned_reviews = [preprocess(review) for review in product_reviews]
X = vectorizer.fit_transform(cleaned_reviews)

# Get word counts
word_counts = np.sum(X.toarray() , axis = 0)
# Get words
words = vectorizer.get_feature_names_out()

top_words_with_stopwords, top_counts_with_stopwords = get_top_ten(words, word_counts)
print(top_words_with_stopwords, top_counts_with_stopwords)
################################
# Modify the function to remove stop words 
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
  
cleaned_reviews = [preprocess(review) for review in product_reviews]
X = vectorizer.fit_transform(cleaned_reviews)

# Get word counts
word_counts = np.sum(X.toarray(), axis=0)
# Get words
words = vectorizer.get_feature_names_out()

top_words_without_stopwords, top_counts_without_stopwords = get_top_ten(words, word_counts)
print(top_words_without_stopwords, top_counts_without_stopwords)
################
import matplotlib.pyplot as plt
# Plot the frequencies with stop words
plt.bar(top_words_with_stopwords , top_counts_with_stopwords)
plt.title("Top 10 word frequencies (with stop words)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

# Plot the frequencies without stop words
plt.figure()
plt.bar(top_words_without_stopwords, top_counts_without_stopwords)
plt.title("Top 10 word frequencies (without stop words)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
#################################
reviews = ["The smart speaker is incredible. Clear sound and fast responses!",
           "I am disappointed with the smart bulb. It stopped working in a week.",
           "The thermostat is okay. Not too smart, but functional."]
cleaned_reviews = [preprocess(review) for review in reviews]

# Initialize the vectorizer
vectorizer = TfidfVectorizer()
# Transform the cleaned reviews
tfidf_matrix = vectorizer.fit_transform(cleaned_reviews)
# Create a DataFrame for TF-IDF
df = pd.DataFrame(
  tfidf_matrix.toarray(),
  columns=vectorizer.get_feature_names_out()
)
print(df.head())
#####################
# Convert BoW matrix to a DataFrame
df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_bow, annot=True)
plt.title("BoW Scores Across Reviews")
plt.xlabel("Terms")
plt.xticks(rotation=45)
plt.ylabel("Documents")
plt.show()
###################################
# Convert TF-IDF matrix to a DataFrame
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_tfidf, annot=True)
plt.title("TF-IDF Scores Across Reviews")
plt.xlabel("Terms")
plt.xticks(rotation=45)
plt.ylabel("Documents")
plt.show()
#############################################
import gensim.downloader as API

Model = API.load("glove-wiki-gigaword-100")
# Compute similarity between "king" and "queen"
similarity_score = model_glove_wiki.similarity("king" , "queen")

print(similarity_score)

# Get top 10 most similar words to "computer"
similar_words = model_glove_wiki.most_similar("computer" , topn = 10)

print(similar_words)