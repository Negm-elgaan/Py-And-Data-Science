# Find the number of positive and negative reviews
print('Number of positive and negative reviews: ', movies.label.value_counts())

# Find the proportion of positive and negative reviews
print('Proportion of positive and negative reviews: ', movies.label.value_counts() / len(movies))
################################
length_reviews = movies.review.str.len()

# How long is the longest review
print(max(length_reviews))
######################################
length_reviews = movies.review.str.len()

# How long is the shortest review
print(min(length_reviews))

#########################################
# Import the required packages
from textblob import TextBlob as TB

# Create a textblob object  
blob_two_cities = TB(two_cities)

# Print out the sentiment 
print(blob_two_cities.sentiment)
################################################
# Import the required packages
from textblob import TextBlob as TB

# Create a textblob object 
blob_annak = TB(annak)
blob_catcher = TB(catcher)

# Print out the sentiment   
print('Sentiment of annak: ', blob_annak.sentiment)
print('Sentiment of catcher: ',blob_catcher.sentiment)
############################
# Import the required packages
from textblob import TextBlob as TB

# Create a textblob object  
blob_titanic = TB(titanic)

# Print out its sentiment  
print(blob_titanic.sentiment)
################################
from wordcloud import WordCloud as WC

# Generate the word cloud from the east_of_eden string
cloud_east_of_eden = WC(background_color="white").generate(east_of_eden)
#####################################
# Generate the word cloud from the east_of_eden string
cloud_east_of_eden = WordCloud(background_color="white").generate(east_of_eden)

# Create a figure of the generated cloud
plt.imshow(cloud_east_of_eden, interpolation='bilinear')  
plt.axis('off')
# Display the figure
plt.show()
################################################
cloud = WordCloud(background_color = "orange").generate(illuminated)
plt.imshow(cloud , interpolation = 'bilinear')
plt.show()
####################################
# Import the word cloud function  
from wordcloud import WordCloud as WC

# Create and generate a word cloud image 
my_cloud = WC(background_color = 'white' , stopwords =my_stopwords).generate(descriptions)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear') 
plt.axis("off")

# Don't forget to show the final image
plt.show()
#############################################
# Import the required function
from sklearn.feature_extraction.text import CountVectorizer as CV

annak = ['Happy families are all alike;', 'every unhappy family is unhappy in its own way']

# Build the vectorizer and fit it
anna_vect = CV()
anna_vect.fit(annak)

# Create the bow representation
anna_bow = anna_vect.transform(annak)

# Print the bag-of-words result 
print(anna_bow.toarray())
###################
from sklearn.feature_extraction.text import CountVectorizer 

# Build the vectorizer, specify max features 
vect = CountVectorizer(max_features=100)
# Fit the vectorizer
vect.fit(reviews.review)

# Transform the review column
X_review = vect.transform(reviews.review)

# Create the bow representation
X_df=pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
##########################################
from sklearn.feature_extraction.text import CountVectorizer 

# Build the vectorizer, specify token sequence and fit
vect = CountVectorizer(ngram_range=(1,2))
vect.fit(reviews.review)

# Transform the review column
X_review = vect.transform(reviews.review)

# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
###########################
from sklearn.feature_extraction.text import CountVectorizer 

# Build the vectorizer, specify size of vocabulary and fit
vect = CountVectorizer(max_features=100)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
###################################
from sklearn.feature_extraction.text import CountVectorizer 

# Build and fit the vectorizer
vect = CountVectorizer(max_df=200)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
########################################
from sklearn.feature_extraction.text import CountVectorizer 

# Build and fit the vectorizer
vect = CountVectorizer(min_df=50)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
###########################################################
#Import the vectorizer
from sklearn.feature_extraction.text import CountVectorizer as CV

# Build the vectorizer, specify max features and fit
vect = CV(max_features=1000, ngram_range=(2, 2), max_df=500)
vect.fit(reviews.review)

# Transform the review
X_review = vect.transform(reviews.review)

# Create a DataFrame from the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
################################
# Import the required function
from nltk import word_tokenize as WT

# Transform the GoT string to word tokens
print(WT(GoT))
#################################################
# Import the word tokenizing function
from nltk import word_tokenize

# Tokenize each item in the avengers 
tokens_avengers = [word_tokenize(item) for item in avengers]

print(tokens_avengers)
################################
# Import the needed packages
from nltk import word_tokenize

# Tokenize each item in the review column 
word_tokens = [word_tokenize(review) for review in reviews.review]

# Print out the first item of the word_tokens list
print(word_tokens[0])
#########################################
# Create an empty list to store the length of reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
reviews['n_words'] = len_tokens 
###############################
# Import the language detection function and package
from langdetect import detect_langs

# Detect the language of the foreign string
print(detect_langs(foreign))
#################################################################
from langdetect import detect_langs

languages = []

# Loop over the sentences in the list and detect their language
for sentence in sentences:
    languages.append(detect_langs(sentence))
    
print('The detected languages are: ', languages)