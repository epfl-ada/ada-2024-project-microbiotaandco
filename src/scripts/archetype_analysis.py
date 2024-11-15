import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

# Data import and Data manipulation
characters = pd.read_csv("src/data/characters_preprocessed_for_archetype.csv")
archetypes = pd.read_csv("src/data/Extracted_Data.csv")
archetypes = archetypes.rename(columns = {'Character_name' : 'Character Name', 'Freebase_movie_ID' : 'Freebase Movie ID', 'Actor_name' : 'Actor Name'})
df = pd.merge(characters, archetypes, how = 'inner', on = ["Freebase Movie ID", "Actor Name"])
df = df.dropna(subset=["Actor Name", "Occupation Labels"])

# One hot encode the occupations
df['Occupation Labels'] = df['Occupation Labels'].str.split(', ')
df_exploded = df.explode('Occupation Labels').reset_index()
occupation_dummies = pd.get_dummies(df_exploded['Occupation Labels'])
df_one_hot_encoded = occupation_dummies.groupby(df_exploded['index']).max()
df_final = df.drop(columns='Occupation Labels').join(df_one_hot_encoded)

# Only keep the occupations with n_per_occupation_treshold or above
n_per_occupation_treshold = 75
dummy_columns = df_one_hot_encoded.columns[df_one_hot_encoded.sum() > n_per_occupation_treshold]
df_filtered = df.drop(columns='Occupation Labels').join(df_one_hot_encoded[dummy_columns])

# Gender
print("Gender in different archetypes:")

# Step 1: Convert 'Actor Gender' to numeric for proportion calculations
df_filtered['Actor Gender Numeric'] = df_filtered['Actor Gender'].map({'M': 1, 'F': 0})

# Step 2: Melt the DataFrame to long format
archetype_columns = df_filtered.columns[21:]  # Adjust to select archetype columns correctly in your dataset
df_melted = df_filtered.melt(id_vars=['Actor Gender', 'Actor Gender Numeric'], 
                    value_vars= archetype_columns,
                    var_name= "Archetype", value_name='Is_Present')

# Step 3: Filter rows where the archetype is present (Is_Present == 1)
df_archetype = df_melted[df_melted['Is_Present'] == 1]

# Step 4: Calculate the mean of 'Actor Gender Numeric' for each archetype
gender_proportion = df_archetype.groupby('Archetype')['Actor Gender Numeric'].mean()

# Step 5: Plot the gender distribution per archetype
plt.figure(figsize=(15, 6))
gender_proportion.plot(kind='bar', color=['skyblue'])
plt.axhline(y=0.5, color='r', linestyle='-')
plt.title("Gender Distribution per Archetype")
plt.ylabel("Proportion of Male Actors")
plt.xlabel("Archetype")
plt.xticks(rotation=90)
plt.ylim(0, 1)  # To indicate proportions from 0 to 1
plt.show()

# Chi2

# Melt the DataFrame to create a long-format DataFrame
df_melted = df_filtered.melt(id_vars=['Actor Gender'], 
                    value_vars=archetype_columns,
                    var_name='Archetype', value_name='Is_Present')

# Filter rows where the archetype is present
df_archetype = df_melted[df_melted['Is_Present'] == 1]

# Create a contingency table for gender and archetype
contingency_table = pd.crosstab(df_archetype['Archetype'], df_archetype['Actor Gender'])

# Perform the Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("Gender statistics:")
print("Chi-square test statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
#print("Expected frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("There is a statistically significant gender bias by archetype (reject null hypothesis).")
else:
    print("No statistically significant gender bias by archetype (fail to reject null hypothesis).")
    
df_filtered = df_filtered.drop(columns = 'Actor Gender Numeric')

#Actors height
print("\nHeight in different archetypes:")


# Extract column names for archetypes (assuming dummy variables start from the 21st column)
archetype_columns = df_filtered.columns[21:]  # Adjust this as needed for your actual DataFrame structure

# Determine common x-axis limits based on the range of all "Actor Height" values
min_height = df_filtered['Actor Height'].min()
max_height = df_filtered['Actor Height'].max()

# Determine grid size based on the number of archetypes
n_archetypes = len(archetype_columns)
n_cols = 3  # Set the number of columns per row in the grid
n_rows = (n_archetypes + n_cols - 1) // n_cols  # Calculate the required number of rows

# Create a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=True)
axes = axes.flatten()  # Flatten axes for easy iteration

# Plot each archetype's height distribution
for i, archetype in enumerate(archetype_columns):
    ax = axes[i]
    heights = df_filtered.loc[df_filtered[archetype] == 1, 'Actor Height']  # Filter heights for the archetype
    ax.hist(heights, bins=10, color='skyblue', edgecolor='black')
    ax.set_title(archetype)
    ax.set_xlabel("Actor Height (m)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(min_height, max_height)  # Set common x-axis limits for consistency

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Height Distribution per Archetype", y=1.05)
plt.tight_layout()
plt.show()

# Chi2

# Melt the DataFrame to create a long-format DataFrame
df_melted = df_filtered.melt(id_vars=['Actor Height'], 
                    value_vars=archetype_columns,
                    var_name='Archetype', value_name='Is_Present')

# Filter rows where the archetype is present
df_archetype = df_melted[df_melted['Is_Present'] == 1]

# Create a contingency table for gender and archetype
contingency_table = pd.crosstab(df_archetype['Archetype'], df_archetype['Actor Height'])

# Perform the Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("Height statistics:")
print("Chi-square test statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
#print("Expected frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("There is a statistically significant height bias by archetype (reject null hypothesis).")
else:
    print("No statistically significant height bias by archetype (fail to reject null hypothesis).")

#Actors age
print("\nAge in different archetypes:")

# Extract column names for archetypes (assuming dummy variables start from the 21st column)
archetype_columns = df_filtered.columns[21:]  # Adjust this as needed for your actual DataFrame structure

# Determine common x-axis limits based on the range of all "Actor Height" values
min_height = df_filtered['Actor Age'].min()
max_height = df_filtered['Actor Age'].max()

# Determine grid size based on the number of archetypes
n_archetypes = len(archetype_columns)
n_cols = 3  # Set the number of columns per row in the grid
n_rows = (n_archetypes + n_cols - 1) // n_cols  # Calculate the required number of rows

# Create a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=True)
axes = axes.flatten()  # Flatten axes for easy iteration

# Plot each archetype's height distribution
for i, archetype in enumerate(archetype_columns):
    ax = axes[i]
    ages = df_filtered.loc[df_filtered[archetype] == 1, 'Actor Age']  # Filter heights for the archetype
    ax.hist(ages, bins=10, color='skyblue', edgecolor='black')
    ax.set_title(archetype)
    ax.set_xlabel("Actor Age")
    ax.set_ylabel("Frequency")
    ax.set_xlim(min_height, max_height)  # Set common x-axis limits for consistency

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Age Distribution per Archetype", y=1.05)
plt.tight_layout()
plt.show()

# Chi2


# Melt the DataFrame to create a long-format DataFrame
df_melted = df_filtered.melt(id_vars=['Actor Age'], 
                    value_vars=archetype_columns,
                    var_name='Archetype', value_name='Is_Present')

# Filter rows where the archetype is present
df_archetype = df_melted[df_melted['Is_Present'] == 1]

#  Create a contingency table for gender and archetype
contingency_table = pd.crosstab(df_archetype['Archetype'], df_archetype['Actor Age'])

# Perform the Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("Age statistics:")
print("Chi-square test statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
#print("Expected frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("There is a statistically significant age bias by archetype (reject null hypothesis).")
else:
    print("No statistically significant age bias by archetype (fail to reject null hypothesis).")

#Actors country of origin
print("\nCountry of origin in different archetypes:")


# Extract column names for archetypes (assuming dummy variables start from the 21st column)
archetype_columns = df_filtered.columns[21:]  # Adjust this as needed for your actual DataFrame structure

# Determine grid size based on the number of archetypes
n_archetypes = len(archetype_columns)
n_cols = 3  # Set the number of columns per row in the grid
n_rows = (n_archetypes + n_cols - 1) // n_cols  # Calculate the required number of rows

# Create a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10 * n_rows), sharey=True)
axes = axes.flatten()  # Flatten axes for easy iteration

# Plot country distributions per archetype
for i, archetype in enumerate(archetype_columns):
    ax = axes[i]
    
    # Filter DataFrame to include only rows where the archetype is present
    countries = df_filtered.loc[df_filtered[archetype] == 1, 'Actor Country of Origin']
    
    # Get the counts of each country
    country_counts = countries.value_counts()
    
    # Plot histogram as a bar chart (since this is categorical data)
    ax.bar(country_counts.index, country_counts.values, color='skyblue', edgecolor='black')
    ax.set_title(archetype)
    ax.set_xlabel("Country of Origin")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=90)  # Rotate x-ticks for readability

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Country of Origin Distribution per Archetype", y=1.05)
plt.tight_layout()
plt.show()

# Chi 2

# Melt the DataFrame to create a long-format DataFrame
df_melted = df_filtered.melt(id_vars=['Actor Country of Origin'], 
                    value_vars=archetype_columns,
                    var_name='Archetype', value_name='Is_Present')

# Filter rows where the archetype is present
df_archetype = df_melted[df_melted['Is_Present'] == 1]

# Create a contingency table for gender and archetype
contingency_table = pd.crosstab(df_archetype['Archetype'], df_archetype['Actor Country of Origin'])

# Perform the Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("Country of origin statistics")
print("Chi-square test statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
#print("Expected frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("There is a statistically significant country of origin by archetype (reject null hypothesis).")
else:
    print("No statistically significant country of origin by archetype (fail to reject null hypothesis).")

#Actors ethnicity
print("\nEthnicity in different archetypes:")


# Extract column names for archetypes (assuming dummy variables start from the 21st column)
archetype_columns = df_filtered.columns[21:]  # Adjust this as needed for your actual DataFrame structure

# Determine grid size based on the number of archetypes
n_archetypes = len(archetype_columns)
n_cols = 3  # Set the number of columns per row in the grid
n_rows = (n_archetypes + n_cols - 1) // n_cols  # Calculate the required number of rows

# Create a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows), sharey=True)
axes = axes.flatten()  # Flatten axes for easy iteration

# Plot ethnicity distributions per archetype
for i, archetype in enumerate(archetype_columns):
    ax = axes[i]
    
    # Filter DataFrame to include only rows where the archetype is present
    ethnicities = df_filtered.loc[df_filtered[archetype] == 1, 'Actor Ethnicity']
    
    # Get the counts of each ethnicity
    ethnicity_counts = ethnicities.value_counts()
    
    # Plot histogram as a bar chart (since this is categorical data)
    ax.bar(ethnicity_counts.index, ethnicity_counts.values, color='skyblue', edgecolor='black')
    ax.set_title(archetype)
    ax.set_xlabel("Ethnicity")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=90)  # Rotate x-ticks for readability

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Ethnicity Distribution per Archetype", y=1.05)
plt.tight_layout()
plt.show()

#Chi 2

# Melt the DataFrame to create a long-format DataFrame
df_melted = df_filtered.melt(id_vars=['Actor Ethnicity'], 
                    value_vars=archetype_columns,
                    var_name='Archetype', value_name='Is_Present')

# Filter rows where the archetype is present
df_archetype = df_melted[df_melted['Is_Present'] == 1]

# Create a contingency table for gender and archetype
contingency_table = pd.crosstab(df_archetype['Archetype'], df_archetype['Actor Ethnicity'])

# Perform the Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("Ethnicity statistics")
print("Chi-square test statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
#print("Expected frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("There is a statistically significant ethnicity bias by archetype (reject null hypothesis).")
else:
    print("No statistically significant ethnicity bias by archetype (fail to reject null hypothesis).")

