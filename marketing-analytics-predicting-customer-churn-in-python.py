telco['Churn'].value_counts()
#####################################
# Group telco by 'Churn' and compute the mean
print(telco.groupby(['Churn']).mean())
###########################
# Adapt your code to compute the standard deviation
print(telco.groupby(['Churn']).std())