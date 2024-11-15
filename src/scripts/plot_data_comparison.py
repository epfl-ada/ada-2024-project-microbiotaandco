import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_column_distributions(df1, df2, column_name):
    """
    Compares the distribution of a column with the same name from two DataFrames using vertical boxplots
    on the same plot.

    Parameters:
    df1 (pandas.DataFrame): First DataFrame.
    df2 (pandas.DataFrame): Second DataFrame.
    column_name (str): The column name to compare.

    Returns:
    None: Displays a single plot with two vertical boxplots side by side.
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

    # Reset index to avoid duplicate labels
    df1_clean = df1_clean.reset_index(drop=True)
    df2_clean = df2_clean.reset_index(drop=True)

    # Combine the data into a single DataFrame
    df_combined = pd.DataFrame({
        'value': pd.concat([df1_clean[column_name], df2_clean[column_name]], ignore_index=True),
        'dataset': ['Archetype Dataset'] * len(df1_clean) + ['Original Character Dataset'] * len(df2_clean)
    })

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dataset', y='value', data=df_combined, orient='v')

    # Set plot title and labels
    plt.title(f'Comparison of {column_name} Distributions')
    plt.ylabel(f'{column_name} Value')
    plt.xlabel('')

    plt.show()


characters = pd.read_csv("df_character_metadata_D.csv")
archetypes = pd.read_csv("../Maurice_M2/final_data.csv")

archetypes = archetypes.rename(columns = {'Character_name' : 'Character Name', 'Freebase_movie_ID' : 'Freebase Movie ID', 'Actor_name' : 'Actor Name'})
df = pd.merge(characters, archetypes, how = 'inner', on = ["Freebase Movie ID", "Actor Name","Character Name"])

df=df.drop_duplicates()
characters.rename(columns={"Movie_release_date": "Movie Release Date"}, inplace=True)
characters=characters[characters["Movie Release Date"]>1800]
compare_column_distributions(df,characters,"Movie Release Date")
compare_column_distributions(df,characters,"Actor Age")
compare_column_distributions(df,characters,"Actor Height")
df["Actor Gender"] = df["Actor Gender"].replace({"F": 1, "M": 0})
# print(df["Actor Gender"].unique())
characters["Actor Gender"]=characters["Actor Gender"].replace({"F": 1, "M": 0})

percent_female_data1 = (df["Actor Gender"].sum() / len(df)) * 100
percent_female_data2 = (characters["Actor Gender"].sum() / len(characters)) * 100

# Plot
datasets = ["Archetype dataset", "Character original dataset"]
percentages = [percent_female_data1, percent_female_data2]

plt.bar(datasets, percentages, color=["#4c72b0", "#4c72b0"])
plt.ylabel("Percentage of Women (%)")
plt.title("Comparison of Women Percentage in Two Datasets")
plt.ylim(0, 100)

for i, v in enumerate(percentages):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center')

