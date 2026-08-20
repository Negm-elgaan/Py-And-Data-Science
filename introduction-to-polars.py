import polars as PL
# Load the CSV into a DataFrame
ev_df = PL.read_csv("electric_vehicles.csv")

# Print the first three rows
print(ev_df.head(3))
###################
# Print number of rows and columns of ev_df
print(ev_df.shape)
##########################
# Print the column names of ev_df
print(ev_df.columns)
################################
# Print the column names and dtypes
print(ev_df.schema)
###########################
# Print the first values in a vertical format
print(ev_df.glimpse())
#################################
# Create a Series from the brand column
print(ev_df['brand'])
####################################
# Extract the brand, model and price columns
print(ev_df[['brand' , 'model' , 'price']])
###############################################
# Print the last three rows of the brand and price columns
print(ev_df[-3 : , ['brand' , 'price']])
########################
# Select the model, accel and price columns
print(ev_df.select(["model" , "accel" , "price"]))
######################################
# Select the brand, model and price columns
print(ev_df.select(["brand" , "model" , "price"]))
############################################
# Sort with longest range first
sorted_ev_df = ev_df.sort("range" , descending = True)

print(sorted_ev_df)
###################################
# Print a summary of the DataFrame
print(ev_df.describe())
###############################
# Get the most expensive 4 vehicles
expensive_df = ev_df.top_k(4 , by = "price")

print(expensive_df)
###########################################
# Find the four cheapest vehicles
cheap_df = ev_df.bottom_k(4  ,  by = "price")

print(cheap_df)