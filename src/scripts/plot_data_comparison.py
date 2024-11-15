import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def compare_column_distributions(df1, df2, column_name, ax):
    """
    Compares the distribution of a column with the same name from two DataFrames using vertical boxplots
    on the same subplot (ax).
    """
    # Ensure both DataFrames contain the specified column
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column '{column_name}' not found in both DataFrames.")
    
    # Check for missing or non-numeric data
    if not pd.api.types.is_numeric_dtype(df1[column_name]):
        raise ValueError(f"Column '{column_name}' in df1 is not numeric.")
    if not pd.api.types.is_numeric_dtype(df2[column_name]):
        raise ValueError(f"Column '{column_name}' in df2 is not numeric.")
    
    # Drop rows with missing values for the selected column
    df1_clean = df1[[column_name]].dropna()
    df2_clean = df2[[column_name]].dropna()

    # Combine the data into a single DataFrame
    df_combined = pd.DataFrame({
        'value': pd.concat([df1_clean[column_name], df2_clean[column_name]], ignore_index=True),
        'dataset': ['Archetype Dataset'] * len(df1_clean) + ['Original Character Dataset'] * len(df2_clean)
    })

    # Create the boxplot
    sns.boxplot(x='dataset', y='value', data=df_combined, orient='v', ax=ax)
    ax.set_title(f'Comparison of {column_name} Distributions', fontsize=10)
    ax.set_ylabel(f'{column_name} Value', fontsize=9)
    ax.set_xlabel('')
    ax.tick_params(axis='both', labelsize=8)

# Load data
characters = pd.read_csv("src/data/characters_preprocessed_for_archetype.csv")
archetypes = pd.read_csv("src/data/Extracted_Data.csv")

archetypes = archetypes.rename(columns = {'Character_name' : 'Character Name', 'Freebase_movie_ID' : 'Freebase Movie ID', 'Actor_name' : 'Actor Name'})
df = pd.merge(characters, archetypes, how = 'inner', on = ["Freebase Movie ID", "Actor Name","Character Name"])
df = df.drop_duplicates()
characters.rename(columns={"Movie_release_date": "Movie Release Date"}, inplace=True)
characters = characters[characters["Movie Release Date"] > 1800]

# Calculate gender percentages
df["Actor Gender"] = df["Actor Gender"].replace({"F": 1, "M": 0})
characters["Actor Gender"] = characters["Actor Gender"].replace({"F": 1, "M": 0})
percent_female_data1 = (df["Actor Gender"].sum() / len(df)) * 100
percent_female_data2 = (characters["Actor Gender"].sum() / len(characters)) * 100

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
compare_column_distributions(df, characters, "Movie Release Date", axes[0, 0])
compare_column_distributions(df, characters, "Actor Age", axes[0, 1])
compare_column_distributions(df, characters, "Actor Height", axes[1, 0])

# Gender percentages bar plot
datasets = ["Archetype Dataset", "Character Original Dataset"]
percentages = [percent_female_data1, percent_female_data2]
axes[1, 1].bar(datasets, percentages, color=["#4c72b0", "#4c72b0"])
axes[1, 1].set_ylabel("Percentage of Women (%)", fontsize=9)
axes[1, 1].set_title("Comparison of Women Percentage in Two Datasets", fontsize=10)
axes[1, 1].set_ylim(0, 100)
axes[1, 1].tick_params(axis='both', labelsize=8)

for i, v in enumerate(percentages):
    axes[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=8)

plt.tight_layout()
plt.show()
