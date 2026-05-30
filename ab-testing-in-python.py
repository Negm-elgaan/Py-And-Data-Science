# Determine the normalized distribution of browser counts
checkout['browser'].value_counts(normalize = True)
#############################
# Determine the normalized distribution of browser counts
checkout['browser'].value_counts(normalize = True)

# Draw a random sample of rows
sample_df = checkout.sample(n = 2000)
##################################
# Determine the normalized distribution of browser counts
checkout['browser'].value_counts(normalize = True)

# Draw a random sample of rows
sample_df = checkout.sample(n = 2000)

# Check the counts distribution of sampled users' browsers
sample_df['browser'].value_counts(normalize = True)
#########################################
# Determine the normalized distribution of browser counts
checkout['browser'].value_counts(normalize = True)

# Draw a random sample of rows
sample_df = checkout.sample(n = 2000)

# Check the counts distribution of sampled users' browsers
sample_df['browser'].value_counts(normalize = True)

# Check the counts distribution of browsers across checkout pages
checkout.groupby('checkout_page')['browser'].value_counts(normalize = True)