import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12 
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

df = pd.read_csv('predictions.csv')

def class_distribution(df):
    tag_list = []

    for tags in df['IOB Slot tags']:
        for tag in tags.split():
            if '_' in tag:
                prefix, category = tag.split('_', 1)
                tag_list.append({'Tag': tag, 'Category': category, 'Prefix': prefix})
            else:
                tag_list.append({'Tag': tag, 'Category': 'O', 'Prefix': 'O'})

    tags_df = pd.DataFrame(tag_list)
    tags_df = pd.DataFrame(tag_list)

    # Remove the 'O' category
    tags_df_filtered = tags_df[tags_df['Category'] != 'O']

    # Count frequencies and group by Category and Prefix (B or I)
    frequency = tags_df_filtered.groupby(['Category', 'Prefix']).size().unstack(fill_value=0)

    # Create the stacked bar plot
    ax = frequency.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10', legend=False)

    # Title and labels (ACL format)
    ax.set_title('Class Distribution of IOB Predictions from Base Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    # Adding the custom legend (B and I)
    handles, labels = ax.get_legend_handles_labels()

    # Map the 'B' and 'I' to custom labels for the legend
    custom_labels = ['B (Beginning)', 'I (Inside)']
    ax.legend(handles=handles[:2], labels=custom_labels, title='Tag Type')

    # Adjust tick labels for readability
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Tight layout for proper spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

print(class_distribution(df))
