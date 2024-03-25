import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("student_data.csv")
#Data Visualization
hue_palette = 'Set2'
# Plot histograms 
for column in df.columns:
    if column != 'G3':
        plt.figure(figsize=(12, 10))
        sns.histplot(data=df, x=column, hue='G3', kde=True, palette=hue_palette)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'{column} vs. G3')
        plt.show()

# Plot countplots 
for column in df.columns:
    if column != 'G3':
        plt.figure(figsize=(12, 12))
        sns.countplot(x=df[column], hue='G3', data=df, palette=hue_palette)
        plt.title(f'{column} vs. G3')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
