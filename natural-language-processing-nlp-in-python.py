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
################################
words = ["lion", "tiger", "leopard", "banana", "strawberry", "truck", "car", "bus"]

# Extract word embeddings
word_vectors = [model_glove_wiki[word] for word in words]

# Reduce dimensions with PCA
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
for word, (x, y) in zip(words, word_vectors_2d):
    plt.annotate(word, (x, y))
plt.title("GloVe Wikipedia Word Embeddings (2D PCA)")
plt.show()
############################################
words = ["lion", "tiger", "leopard", "banana", "strawberry", "truck", "car", "bus"]

# Change the embedding model
word_vectors = [model_glove_twitter[word] for word in words]

# Reduce dimensions with PCA
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
for word, (x, y) in zip(words, word_vectors_2d):
    plt.annotate(word, (x, y))
plt.title("GloVe Twitter Word Embeddings (2D PCA)")
plt.show()
############################
from transformers import pipeline as pl

# Define the sentiment analysis pipeline
classifier = pl(task = "sentiment-analysis" , model = "distilbert-base-uncased-finetuned-sst-2-english")

review_text = "The new update made the app much faster and easier to use!"

# Get sentiment prediction
result = classifier(review_text)

print(result)
######################################
from transformers import pipeline

classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

review_batch = [
    "Absolutely love the new design!",
    "The app crashes every time I open it.",
    "Customer support was helpful and quick.",
    "Too many ads make it unusable.",
    "Everything works fine, but it’s a bit slow."
]

# Classify sentiments
results = classifier(review_batch)
print(results)
##############################################
from transformers import pipeline
from sklearn.metrics import accuracy_score
# Load sentiment analysis models
pipe_a = pipeline(task="sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")
pipe_b = pipeline(task="sentiment-analysis", model = "abilfad/sentiment-binary-dicoding")

# Generate predictions
preds_a = [res['label'] for res in pipe_a(texts)]
preds_b = [res['label'] for res in pipe_b(texts)]
##########################################################
from transformers import pipeline
from sklearn.metrics import accuracy_score
# Load sentiment analysis models
pipe_a = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
pipe_b = pipeline(task="sentiment-analysis", model="abilfad/sentiment-binary-dicoding")

# Generate predictions
preds_a = [res["label"] for res in pipe_a(texts)]
preds_b = [res["label"] for res in pipe_b(texts)]

# Evaluate accuracies
acc_a = accuracy_score(preds_a , true_labels)
acc_b = accuracy_score(preds_b , true_labels)
print(f"Accuracy - Model A: {acc_a:.2f}")
print(f"Accuracy - Model B: {acc_b:.2f}")
##################################################################
from transformers import pipeline

# Initialize the zero-shot classifier
classifier = pipeline(task = "zero-shot-classification" , model = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

ticket_text = "I was charged twice for my subscription this month. Can you please refund the extra charge?"
candidate_labels = ["Billing", "Technical Issue", "Account Access"]

# Classify the ticket
result = classifier(ticket_text , candidate_labels)

print(result['labels'])
print(result['scores'])
####################################
from transformers import pipeline

# Initialize the QNLI pipeline
classifier = pipeline(model = "cross-encoder/qnli-electra-base" , task = "text-classification")

passage = "Our refund policy allows customers to return any item within 30 days of purchase, provided the item is in its original condition and accompanied by the receipt. Refunds are issued to the original payment method within 5–7 business days."
question = "Can I get a refund if I return a product after 20 days?"

# Get the result
result = classifier({"text" : question , "text_pair": passage})
print(result)
###########################################
from transformers import pipeline

# Initialize the pipeline
classifier = pipeline(task="text-classification", model="textattack/bert-base-uncased-QQP")

question_1 = "What's the process to change my password?"
question_2 = "How do I reset my account password?"

# Detect if the two questions are paraphrases
result = classifier({
    "text": question_1,
    "text_pair": question_2
})

print(result)
###################################
from transformers import pipeline

# Initialize the pipeline
classifier = pipeline(task="text-classification", model="textattack/distilbert-base-uncased-QQP")

question_1 = "What's the process to change my password?"
question_2 = "How do I reset my account password?"

# Detect if the two questions are paraphrases
result = classifier({
    "text": question_1,
    "text_pair": question_2
})

print(result)
###################################################
from transformers import pipeline

# Initialize the pipeline
classifier = pipeline(task = "text-classification" , model = "textattack/bert-base-uncased-CoLA")

user_text = "Although she was knowing the answer, she didn't raised her hand during the class discussion."

# Classify grammatical acceptability
result = classifier(user_text)

print(result)
###########################
from transformers import pipeline
# Create the NER pipeline
ner_pipeline = pipeline(
    task="ner",
    model="dslim/bert-base-NER",
    grouped_entities=True
)
headline = "Apple is planning to open a new office in San Francisco next year."

# Get named entities
entities = ner_pipeline(headline)

for entity in entities:
    print(f"{entity['entity_group']}: {entity['word']}")
######################################
from transformers import pipeline
# Create the PoS tagging pipeline
pos_pipeline = pipeline(
    task="token-classification",
    model="vblagoje/bert-english-uncased-finetuned-pos",
    grouped_entities=True
)

sentence = "I am meeting my friends for coffee this afternoon."

# Get PoS tags
pos_tags = pos_pipeline(sentence)
for token in pos_tags:
    print(f"{token['word']}: {token['entity_group']}")
################################################
from transformers import pipeline

# Create the question-answering pipeline
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

context = """This smartphone features a 6.5-inch OLED display, 128GB of storage, and a 48MP camera with night mode. It supports 5G connectivity and has a battery life of up to 24 hours."""

question = "What is the size of the smartphone's display?"

# Get the answer
result = qa_pipeline(question , context)
print(result)
##################################################################################################
from transformers import pipeline

# Create the abstractive question-answering pipeline
qa_pipeline = pipeline(
    task="text2text-generation",
    model="fangyuan/hotpotqa_abstractive"
)

context = """This smartphone features a 6.5-inch OLED display, 128GB of storage, and a 48MP camera with night mode. It supports 5G connectivity and has a battery life of up to 24 hours."""

question = "What is the size of the smartphone's display?"

# Generate abstractive answer
result = qa_pipeline(f"question: {question} context: {context}")
print(result)
####################################
from transformers import pipeline

# Create the summarization pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

article = """NASA's Perseverance rover has successfully collected its first rock samples from Mars, marking a significant milestone in the mission. The samples will be stored for potential return to Earth in the future, providing valuable insight into the planet's geology and potential signs of past microbial life."""

# Generate the summary
summary = summarizer(article)

print(summary)
##############
from transformers import pipeline

# Create the translation pipeline
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")

review = "The hotel was clean and the staff were very friendly."

# Translate the review
translation = translator(review)

print(translation)
################################################################
from transformers import pipeline

# Create the translation pipeline
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")

review = "The hotel was clean and the staff were very friendly."

# Translate the review
translation = translator(review)

print(translation)