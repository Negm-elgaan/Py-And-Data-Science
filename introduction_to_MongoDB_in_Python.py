# Import MongoClient
from pymongo import MongoClient as MC
# Create client
client = MC()
################################
from pymongo import MongoClient 
client = MongoClient()

# Print out the names of all databases in your server
print(client.list_database_names())

# Print out the names of all collection in the film database
print(client.film.list_collection_names())
#######################################
from pymongo import MongoClient 
client = MongoClient()

# Create mov
mov = client.film.movies

# Fetch all movies in the collection
all_movies = list(mov.find())
print(f"Retrieved {len(all_movies)} movies")
print(all_movies)

# Fetch all movies that have a release year of 2008
some_movies = list(mov.find({'release_year':2008}))
print(f"Retrieved {len(some_movies)} movies:")
print(some_movies)
######################################################
# Find the movie with the title "her"
her = mov.find_one({'title':"her"})
print(her)

# Find a movie with "genre" of "adventure"
adv = mov.find_one({'genre':"adventure"})
print(adv)
###########################
# Find movie that has release_year 2010 or 2011
tens = mov.find_one({"release_year" : {"$in":[2010 , 2011]}})
print(tens)

# Find movie that has genre "action" or "comedy" 
bang = mov.find_one({"genre" : {"$in" : ["action" , "comedy"]}})
print(bang)