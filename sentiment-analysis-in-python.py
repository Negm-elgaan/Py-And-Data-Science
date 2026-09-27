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