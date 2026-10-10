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