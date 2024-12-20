import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import bar_chart_race as bcr
import scipy.stats as stats


# import data
characters = pd.read_csv("src/data/characters_preprocessed_for_archetype.csv")
archetypes = pd.read_csv("src/data/Extracted_Data.csv")

archetypes = archetypes.rename(columns = {'Character_name' : 'Character Name', 'Freebase_movie_ID' : 'Freebase Movie ID', 'Actor_name' : 'Actor Name'})
df = pd.merge(characters, archetypes, how = 'inner', on = ["Freebase Movie ID", "Actor Name"])

#drop NAs in occupations labels and actor names
df = df.dropna(subset=["Actor Name", "Occupation Labels"])

#one hot encode the occupations
#Split the comma-separated strings into lists
df['Occupation Labels'] = df['Occupation Labels'].str.split(', ')

#Explode the lists so each occupation is in a separate row
df_exploded = df.explode('Occupation Labels').reset_index()  # Reset index to retain original row identifiers

#One-hot encode the 'Occupation Labels' column
occupation_dummies = pd.get_dummies(df_exploded['Occupation Labels'])

#Group by the original 'index' column and aggregate
df_one_hot_encoded = occupation_dummies.groupby(df_exploded['index']).max()

#Combine with the original DataFrame (excluding the exploded column to avoid duplication)
df_final = df.drop(columns='Occupation Labels').join(df_one_hot_encoded)

#all archetypes above 50
dummy_columns = df_one_hot_encoded.columns[df_one_hot_encoded.sum() > 50]
df_filtered = df.drop(columns='Occupation Labels').join(df_one_hot_encoded[dummy_columns])

# Bar chart races plots:

df_filtered['Movie Release Date'] = df_filtered['Movie Release Date'].astype(str).str.replace('.0', '', regex=False)
df_filtered['Movie Release Date'] = pd.to_datetime(df_filtered['Movie Release Date'], format='%Y')
df_filtered.set_index('Movie Release Date', inplace=True)

# Select the one-hot encoded archetype columns (assuming they are from index 21 onward)
archetype_columns = df_filtered.columns[21:]

# Resample the data by year
df_resampled = df_filtered[archetype_columns].resample('YE').sum()

# Sum of previous 10 years
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative, filename='archetypes_bar_chart_race.mp4',title='Archetypes over Time',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)


# Filter the data to include only rows where Actor Ethnicity is 'Jewish'
df_filtered_jewish = df_filtered[df_filtered['Actor Ethnicity'] == "['Jewish']"]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_jewish.columns[21:]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_jewish.columns[21:]

# Resample the data by year 
df_resampled = df_filtered_jewish[archetype_columns].resample('YE').sum()
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative,filename='archetypes_bar_chart_race_jewish.mp4',title='Archetypes over Time for jewish ethnicity',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)

# Filter the data to include only rows where Actor Ethnicity is 'European Americans'
df_filtered_euro_am = df_filtered[df_filtered['Actor Ethnicity'] == "['European Americans']"]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_euro_am.columns[21:]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_euro_am.columns[21:]

# Resample the data by year 
df_resampled = df_filtered_euro_am[archetype_columns].resample('YE').sum()
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative,filename='archetypes_bar_chart_race_euro_american.mp4',title='Archetypes over Time for european american ethnicity',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)


# Filter the data to include only rows where Actor Ethnicity is 'British'
df_filtered_brit = df_filtered[df_filtered['Actor Ethnicity'] == "['British']"]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_brit.columns[21:]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_brit.columns[21:]

# Resample the data by year 
df_resampled = df_filtered_brit[archetype_columns].resample('YE').sum()
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative,filename='archetypes_bar_chart_race_british.mp4',title='Archetypes over Time for british ethnicity',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)


# Filter the data to include only rows where Actor Ethnicity is 'African Americans'
df_filtered_african_am = df_filtered[df_filtered['Actor Ethnicity'] == "['African Americans']"]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_african_am.columns[21:]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_african_am.columns[21:]

# Resample the data by year 
df_resampled = df_filtered_african_am[archetype_columns].resample('YE').sum()
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative,filename='archetypes_bar_chart_race_african_american.mp4',title='Archetypes over Time for african american ethnicity',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)



# Filter the data to include only rows where Actor gender is 'Female'
df_filtered_f = df_filtered[df_filtered['Actor Gender'] == "F"]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_f.columns[21:]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_f.columns[21:]

# Resample the data by year 
df_resampled = df_filtered_f[archetype_columns].resample('YE').sum()
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative,filename='archetypes_bar_chart_race_female.mp4',title='Archetypes over Time for females',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)



# Filter the data to include only rows where Actor gender is 'Male'
df_filtered_m = df_filtered[df_filtered['Actor Gender'] == "M"]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_m.columns[21:]

# Select the one-hot encoded archetype columns
archetype_columns = df_filtered_m.columns[21:]

# Resample the data by year 
df_resampled = df_filtered_m[archetype_columns].resample('YE').sum()
df_cumulative = df_resampled[archetype_columns].rolling(window=10, min_periods=1).sum()

# Create the bar chart race (uncomment to create graph)
#bcr.bar_chart_race(df=df_cumulative,filename='archetypes_bar_chart_race_male.mp4',title='Archetypes over Time for males',steps_per_period=10,
#    period_length=500, figsize=(8, 6), n_bars = 6, period_label={'x': 1, 'y': 0, 'ha': 'right', 'va': 'bottom'}, period_fmt='%Y',)

#chi2 test for ethnicity for the 4 most represented ones
df = df_filtered[(df_filtered['Actor Ethnicity'] == "['Jewish']") | 
                 (df_filtered['Actor Ethnicity'] == "['European Americans']") | 
                 (df_filtered['Actor Ethnicity'] == "['British']") | 
                 (df_filtered['Actor Ethnicity'] == "['African Americans']")]

# Create a contingency table for all archetypes combined
contingency_table = pd.crosstab(df['Actor Ethnicity'], [df[col] for col in df.columns[21:]])

# Perform the chi2 for ethnicities
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistics for Ethnicities across all archetypes:")
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("-" * 100)

#chi2 per archetype for ethnicities
results = []
for archetype in df_filtered.columns[21:]: 
    contingency_table = pd.crosstab(df_filtered['Actor Ethnicity'], df_filtered[archetype])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    results.append((archetype, chi2, p))

# Sort the results by p-value in ascending order
sorted_results = sorted(results, key=lambda x: x[2])

# Print the top 5 most significant results (smallest p-values)
print("Top 5 Most Significant ethnicity disparate archetypes Chi-Square Tests:")
print("-" * 50)
for archetype, chi2, p in sorted_results[:5]:
    print(f"Chi-Square Test for '{archetype}':")
    print(f"Chi-Square Statistic: {chi2:.2f}")
    print(f"P-value: {p:.2e}")
    print("-" * 50)

#chi2 test for gender
# Create a contingency table for all archetypes combined
contingency_table = pd.crosstab(df_filtered['Actor Gender'], [df_filtered[col] for col in df_filtered.columns[21:]])

# Perform the Chi-Square Test for Homogeneity
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistics for Genders across all archetypes:")
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("-" * 100)



#chi2 per archetype for genders
results = []

for archetype in df_filtered.columns[21:]:
    contingency_table = pd.crosstab(df_filtered['Actor Gender'], df_filtered[archetype])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    results.append((archetype, chi2, p))

# Sort the results by p-value in ascending order
sorted_results = sorted(results, key=lambda x: x[2])

# Print the top 5 most significant results (smallest p-values)
print("Top 5 Most Significant gender disparate archetypes Chi-Square Tests:")
print("-" * 50)
for archetype, chi2, p in sorted_results[:5]:
    print(f"Chi-Square Test for '{archetype}':")
    print(f"Chi-Square Statistic: {chi2:.2f}")
    print(f"P-value: {p:.5e}")
    print("-" * 50)









