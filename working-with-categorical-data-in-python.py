# Explore the Above/Below 50k variable
print(adult['Above/Below 50k'].describe())
#####################
# Explore the Above/Below 50k variable
print(adult["Above/Below 50k"].describe())

# Print a frequency table of "Above/Below 50k"
print(adult["Above/Below 50k"].value_counts())
####################
# Explore the Above/Below 50k variable
print(adult["Above/Below 50k"].describe())

# Print a frequency table of "Above/Below 50k"
print(adult["Above/Below 50k"].value_counts())

# Print relative frequency values
print(adult["Above/Below 50k"].value_counts(normalize = True))
#############################
# Create a Series, default dtype
series1 = pd.Series(list_of_occupations)
###################
# Create a Series, default dtype
series1 = pd.Series(list_of_occupations)

# Print out the data type and number of bytes for series1
print("series1 data type:", series1.dtype)
print("series1 number of bytes:", series1.nbytes)
###########################
# Create a Series, default dtype
series1 = pd.Series(list_of_occupations)

# Print out the data type and number of bytes for series1
print("series1 data type:", series1.dtype)
print("series1 number of bytes:", series1.nbytes)

# Create a Series, "category" dtype
series2 = pd.Series(list_of_occupations, dtype='category')
#############################
# Create a Series, default dtype
series1 = pd.Series(list_of_occupations)

# Print out the data type and number of bytes for series1
print("series1 data type:", series1.dtype)
print("series1 number of bytes:", series1.nbytes)

# Create a Series, "category" dtype
series2 = pd.Series(list_of_occupations, dtype="category")

# Print out the data type and number of bytes for series2
print("series2 data type:", series2.dtype)
print("series2 number of bytes:", series2.nbytes)
###########################
# Create a categorical Series and specify the categories (let pandas know the order matters!)
medals = pd.Categorical(medals_won , categories = ["Bronze" , "Silver" , "Gold"] , ordered = True )
print(medals)
######################
# Check the dtypes
print(adult.dtypes)
########################
# Check the dtypes
print(adult.dtypes)

# Create a dictionary with column names as keys and "category" as values
adult_dtypes = {
  "Workclass" : "category" , "Education" : "category", "Relationship" : "category" , "Above/Below 50k" : "category"
}
#######################
# Check the dtypes
print(adult.dtypes)

# Create a dictionary with column names as keys and "category" as values
adult_dtypes = {
   "Workclass": "category",
   "Education": "category",
   "Relationship": "category",
   "Above/Below 50k": "category" 
}

# Read in the CSV using the dtypes parameter
adult2 = pd.read_csv(
  "adult.csv",
  dtype = adult_dtypes
)
print(adult2.dtypes)
######################
# Group the adult dataset by "Sex" and "Above/Below 50k"
gb = adult.groupby(by = ['Sex',"Above/Below 50k"])

# Print out how many rows are in each created group
print(gb.size())

# Print out the mean of each group for all columns
print(gb.mean())
##############################
# Check frequency counts while also printing the NaN count
print(dogs["keep_in"].value_counts(dropna = False))
##############################
# Check frequency counts while also printing the NaN count
print(dogs["keep_in"].value_counts(dropna=False))

# Switch to a categorical variable
dogs["keep_in"] = dogs['keep_in'].astype('category')
################
# Check frequency counts while also printing the NaN count
print(dogs["keep_in"].value_counts(dropna=False))

# Switch to a categorical variable
dogs["keep_in"] = dogs["keep_in"].astype("category")

# Add new categories
new_categories = ["Unknown History", "Open Yard (Countryside)"]
dogs["keep_in"] = dogs["keep_in"].cat.add_categories(new_categories)
##################
# Check frequency counts while also printing the NaN count
print(dogs["keep_in"].value_counts(dropna=False))

# Switch to a categorical variable
dogs["keep_in"] = dogs["keep_in"].astype("category")

# Add new categories
new_categories = ["Unknown History", "Open Yard (Countryside)"]
dogs["keep_in"] = dogs["keep_in"].cat.add_categories(new_categories)

# Check frequency counts one more time
print(dogs["keep_in"].value_counts(dropna=False))
##########################
# Set "maybe" to be "no"
dogs.loc[dogs["likes_children"] == "maybe", "likes_children"] = "no"

# Print out categories
print(dogs["likes_children"].cat.categories)
########################
# Set "maybe" to be "no"
dogs.loc[dogs["likes_children"] == "maybe", "likes_children"] = "no"

# Print out categories
print(dogs["likes_children"].cat.categories)

# Print the frequency table
print(dogs["likes_children"].value_counts(dropna = False))
#########################
# Set "maybe" to be "no"
dogs.loc[dogs["likes_children"] == "maybe", "likes_children"] = "no"

# Print out categories
print(dogs["likes_children"].cat.categories)

# Print the frequency table
print(dogs["likes_children"].value_counts())

# Remove the "maybe" category
dogs["likes_children"] = dogs["likes_children"].cat.remove_categories(["maybe"])
print(dogs["likes_children"].value_counts())
################################
# Set "maybe" to be "no"
dogs.loc[dogs["likes_children"] == "maybe", "likes_children"] = "no"

# Print out categories
print(dogs["likes_children"].cat.categories)

# Print the frequency table
print(dogs["likes_children"].value_counts())

# Remove the `"maybe" category
dogs["likes_children"] = dogs["likes_children"].cat.remove_categories(["maybe"])
print(dogs["likes_children"].value_counts())

# Print the categories one more time
print(dogs["likes_children"].cat.categories)
#####################################
# Create the my_changes dictionary
my_changes = {'Maybe?' : 'Maybe'}

# Rename the categories listed in the my_changes dictionary
dogs["likes_children"] = dogs['likes_children'].cat.rename_categories(new_categories = my_changes)

# Use a lambda function to convert all categories to uppercase using upper()
dogs["likes_children"] =  dogs["likes_children"].cat.rename_categories(lambda c: c.upper())

# Print the list of categories
print(dogs["likes_children"].cat.categories)
############################
# Create the update_coats dictionary
update_coats = {'wirehaired' : 'medium' , 'medium-long' : 'medium'}

# Create a new column, coat_collapsed
dogs["coat_collapsed"] = dogs["coat"].replace(update_coats)

# Convert the column to categorical
dogs["coat_collapsed"] = dogs["coat_collapsed"].astype('category')

# Print the frequency table
print(dogs["coat_collapsed"].value_counts())
#####################
# Print out the current categories of the size variable
print(dogs['size'].cat.categories)
#################
# Print out the current categories of the size variable
print(dogs["size"].cat.categories)

# Reorder the categories using the list provided
dogs["size"] = dogs['size'].cat.reorder_categories(new_categories = ["small", "medium", "large"])
########################
# Print out the current categories of the size variable
print(dogs["size"].cat.categories)

# Reorder the categories, specifying the Series is ordinal
dogs["size"] = dogs["size"].cat.reorder_categories(
  new_categories=["small", "medium", "large"],
  ordered = True
)
##########################
# Print out the current categories of the size variable
print(dogs["size"].cat.categories)

# Reorder the categories, specifying the Series is ordinal, and overwriting the original series
dogs["size"].cat.reorder_categories(
  new_categories=["small", "medium", "large"],
  ordered=True,
  inplace = True
)
#############################
# Previous code
dogs["size"].cat.reorder_categories(
  new_categories=["small", "medium", "large"],
  ordered=True,
  inplace=True
)

# How many Male/Female dogs are available of each size?
print(dogs.groupby(by = ['size'])['sex'].value_counts())

# Do larger dogs need more room to roam?
print(dogs.groupby(by = ['size'])["keep_in"].value_counts())
#######################
# Fix the misspelled word 
replace_map = {"Malez" : 'male'}
##############################
# Fix the misspelled word 
replace_map = {"Malez": "male"}

# Update the sex column using the created map
dogs["sex"] = dogs['sex'].replace(replace_map)

print(dogs["sex"].value_counts())
#######################
# Fix the misspelled word
replace_map = {"Malez": "male"}

# Update the sex column using the created map
dogs["sex"] = dogs["sex"].replace(replace_map)

# Strip away leading whitespace
dogs["sex"] = dogs['sex'].str.strip()

print(dogs["sex"].value_counts())
################
# Fix the misspelled word
replace_map = {"Malez": "male"}

# Update the sex column using the created map
dogs["sex"] = dogs["sex"].replace(replace_map)

# Strip away leading whitespace
dogs["sex"] = dogs["sex"].str.strip()

# Make all responses lowercase
dogs["sex"] = dogs['sex'].str.lower()

print(dogs["sex"].value_counts())
####################
# Fix the misspelled word
replace_map = {"Malez": "male"}

# Update the sex column using the created map
dogs["sex"] = dogs["sex"].replace(replace_map)

# Strip away leading whitespace
dogs["sex"] = dogs["sex"].str.strip()

# Make all responses lowercase
dogs["sex"] = dogs["sex"].str.lower()

# Convert to a categorical Series
dogs["sex"] = dogs['sex'].astype('category')

print(dogs["sex"].value_counts())
##################
# Print the category of the coat for ID 23807
print(dogs.loc[23807 , 'coat'])
#######################
# Find the count of male and female dogs who have a "long" coat
print(dogs.loc[dogs['coat'] == 'long', 'sex'].value_counts())
###########################
# Print the mean age of dogs with a breed of "English Cocker Spaniel"
print(dogs.loc[dogs["breed"] == "English Cocker Spaniel" , 'age'].mean())
##########################
# Count the number of dogs that have "English" in their breed name
print(dogs[dogs["breed"].str.contains('English', regex=False)].shape[0])
#########################
# Set the font size to 1.25
sns.set(font_scale = 1.25)
# Set the background to "darkgrid"
sns.set_style('darkgrid')

# Create a boxplot
sns.catplot(kind = 'box' , y = 'Helpful votes' , x = 'Traveler type' , data = reviews)

plt.show()
############################
# Print the frequency counts of "Period of stay"
print(reviews['Period of stay'].value_counts())

sns.set(font_scale=1.4)
sns.set_style("whitegrid")

# Create a bar plot of "Helpful votes" by "Period of stay"
sns.catplot(kind = 'bar' , y = "Helpful votes" , x = "Period of stay" , data = reviews)
plt.show()
###############################
# Create a bar chart
sns.set(font_scale=.9)
sns.set_style("whitegrid")
sns.catplot(x = "User continent", y = "Score", data=reviews, kind= 'bar')
plt.show()
#################
# Create a bar chart
sns.set(font_scale=.9)
sns.set_style("whitegrid")
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")

# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())
###########################
# Set style
sns.set(font_scale=.9)
sns.set_style("whitegrid")

# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())

# Convert "User continent" to a categorical variable
reviews["User continent"] = reviews["User continent"].astype('category')
sns.catplot(x = 'User continent' , y="Score", data=reviews, kind="bar")
plt.show()
#######################
# Set style
sns.set(font_scale=.9)
sns.set_style("whitegrid")

# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())

# Convert "User continent" to a categorical variable
reviews["User continent"] = reviews["User continent"].astype("category")

# Reorder "User continent" using continent_categories and rerun the graphic
continent_categories = list(reviews["User continent"].value_counts().index)
reviews["User continent"] = reviews["User continent"].cat.reorder_categories(new_categories = continent_categories)
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")
plt.show()
##########################
# Add a second category to split the data on: "Free internet"
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x = "Casino" , y="Score", data=reviews, kind="bar", hue = "Free internet")
plt.show()
##############################
# Switch the x and hue categories
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x = "Free internet", y="Score", data=reviews, kind="bar", hue= "Casino")
plt.show()
###############################
# Update x to be "User continent"
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x = "User continent", y="Score", data=reviews, kind="bar", hue="Casino")
plt.show()
#######################
# Lower the font size so that all text fits on the screen.
sns.set(font_scale = 1.0)
sns.set_style("darkgrid")
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar", hue="Casino")
plt.show()
#####################
# Create a point plot with catplot using "Hotel stars" and "Nr. reviews"
sns.catplot(
  # Split the data across Hotel stars and summarize Nr. reviews
  x = "Hotel stars",
  y = "Nr. reviews",
  data=reviews,
  # Specify a point plot
  kind = 'point',
  hue= "Pool",
  # Make sure the lines and points don't overlap
  dodge = True , 
  join = False
)
plt.show()
####################
sns.set(font_scale=1.4)
sns.set_style("darkgrid")

# Create a catplot that will count the frequency of "Score" across "Traveler type"
sns.catplot(
  kind = 'count' ,
  x = 'Score' ,
  hue = 'Traveler type' ,
  data  = reviews
)
plt.show()
#####################
# Create a catplot for each "Period of stay" broken down by "Review weekday"
ax = sns.catplot(
  # Make sure Review weekday is along the x-axis
  x = "Review weekday",
  # Specify Period of stay as the column to create individual graphics for
  col = "Period of stay" ,
  # Specify that a count plot should be created
  kind = 'count',
  # Wrap the plots after every 2nd graphic.
  col_wrap = 2,
  data=reviews
)
plt.show()