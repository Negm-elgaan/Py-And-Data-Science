# Import the necessary functions
from torchtext.data.utils import get_tokenizer
from nltk.probability import FreqDist 

text = "In the city of Dataville, a data analyst named Alex explores hidden insights within vast data. With determination, Alex uncovers patterns, cleanses the data, and unlocks innovation. Join this adventure to unleash the power of data-driven decisions."

# Initialize the tokenizer and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

threshold = 1
# Remove rare words and print common tokens
freq_dist = FreqDist(tokens)
common_tokens = [token for token in tokens if freq_dist[token] > 1]
print(common_tokens)
####################
# Initialize and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)
print(tokens)
############################
# Initialize and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

# Remove any stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]
print(filtered_tokens)
#######################################
# Initialize and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

# Remove any stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Perform stemming on the filtered tokens
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print(stemmed_tokens)
##################################################
genres = ['Fiction','Non-fiction','Biography', 'Children','Mystery']

# Define the size of the vocabulary
vocab_size = len(genres)

# Create one-hot vectors
one_hot_vectors = torch.eye(vocab_size)

# Create a dictionary mapping genres to their one-hot vectors
one_hot_dict = {genre : one_hot_vectors[i] for i, genre in enumerate(genres)}

for genre, vector in one_hot_dict.items():
    print(f'{genre}: {vector.numpy()}')
#############################################################
# Import from sklearn
from sklearn.feature_extraction.text import CountVectorizer

titles = ['The Great Gatsby','To Kill a Mockingbird','1984','The Catcher in the Rye','The Hobbit', 'Great Expectations']

# Initialize Bag-of-words with the list of book titles
vectorizer = CountVectorizer()
bow_encoded_titles = vectorizer.fit_transform(titles)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(bow_encoded_titles.toarray()[0, :5])
#####################################################################
# Importing TF-IDF from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF encoding vectorizer
vectorizer = TfidfVectorizer()
tfidf_encoded_descriptions = vectorizer.fit_transform(descriptions)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(tfidf_encoded_descriptions.toarray()[0, :5])
########################
# Create a list of stopwords
stop_words = set(stopwords.words("english"))
#################################
# Create a list of stopwords
stop_words = set(stopwords.words("english"))

# Initialize the tokenizer and stemmer
tokenizer = get_tokenizer("basic_english")
stemmer = PorterStemmer() 
############################
# Create a list of stopwords
stop_words = set(stopwords.words("english"))

# Initialize the tokenizer and stemmer
tokenizer = get_tokenizer("basic_english")
stemmer = PorterStemmer() 

# Complete the function to preprocess sentences
def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
		# Tokenize the sentence
        tokens = tokenizer(sentence)
		# Remove stop words
        tokens = [token for token in tokens if token not in stop_words]
		# Stem the tokens
        tokens = [stemmer.stem(token) for token in tokens]
        processed_sentences.append(' '.join(tokens))
    return processed_sentences

processed_shakespeare = preprocess_sentences(shakespeare)
print(processed_shakespeare[:5]) 
################################
# Define your Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
#############################
# Define your Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data) 
    def __getitem__(self, idx):
        return self.data[idx]

# Complete the encoding function
def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer
#####################
# Define your Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Complete the encoding function
def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer
    
# Complete the text processing pipeline
def text_processing_pipeline(sentences):
    processed_sentences = preprocess_sentences(sentences)
    encoded_sentences, vectorizer = encode_sentences(processed_sentences)
    dataset = ShakespeareDataset(encoded_sentences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer

dataloader, vectorizer = text_processing_pipeline(processed_shakespeare)

# Print the vectorizer's feature names and the first 10 components of the first item
print(vectorizer.get_feature_names_out()[:10]) 
print(next(iter(dataloader))[0, :10])