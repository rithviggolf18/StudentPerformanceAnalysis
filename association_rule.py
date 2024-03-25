import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Read the dataset from a CSV file
data = pd.read_csv("student_data.csv")

# Map "yes" to 1 and "no" to 0 for the selected features
binary_features = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
data[binary_features] = data[binary_features].replace({'yes': 1, 'no': 0})

# Map G3 values to binary (0 or 1)
data['G3_binary'] = data['G3'].apply(lambda x: 0 if x <= 11 else 1)

# Select the features for association rules mining
association_data = data[binary_features + ['G3_binary']]

# Perform frequent itemset generation using Apriori algorithm
frequent_itemsets = apriori(association_data, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Display the association rules
# Sort the association rules DataFrame by support
sorted_rules_by_support = rules.sort_values(by='support', ascending=False)

# Print the top 10 rules sorted by support
print(sorted_rules_by_support[['antecedents', 'consequents', 'support', 'confidence']].head(20))