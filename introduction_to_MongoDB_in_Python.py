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