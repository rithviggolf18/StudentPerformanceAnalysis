import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_data.csv")

# Remove non-numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
df_numeric = df[numeric_columns]

correlation_matrix = df_numeric.corr()
plt.figure(figsize=(15, 15))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt=".2f")
plt.title('Correlation Matrix (Numerical Variables)')

# Increase space between y-axis labels
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)  
plt.show()

# Order features in descending order of correlation coefficient 
correlation_with_g3 = correlation_matrix['G3'].abs().sort_values(ascending=False)
print("Correlation with G3:")
print(correlation_with_g3)
