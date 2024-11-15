import pandas as pd
import numpy as np
import ast
import os

df_character_metadata = pd.read_csv("src/data/character_metadata.csv")
df_movie_metadata = pd.read_csv("src/data/movie_metadata.csv")

df_character_metadata_A = df_character_metadata[['Freebase Movie ID', 'Movie Release Date', 'Character Name','Actor Name', 'Actor Gender', 'Actor Height', 'Actor Ethnicity', 'Actor age at movie realise']]

df_character_metadata_B = df_character_metadata_A

# Conveting the format
df_charcter_movie_release_date = df_character_metadata_B['Movie Release Date'].str[:4]
# Convert the Series directly to numeric
df_charcter_movie_release_date = pd.to_numeric(df_charcter_movie_release_date, errors='coerce')

# Create a new DataFrame with specific columns
df_actor_height = df_character_metadata_B[['Actor Name', 'Actor Height']].copy()

def change_height(data_raw):
    '''Correct the height according to the actor's name'''
    if data_raw['Actor Name'] == 'Zohren Weiss':
        data_raw['Actor Height'] = np.nan
    elif data_raw['Actor Name'] == 'Vince Corazza':
        data_raw['Actor Height'] = 1.78
    elif data_raw['Actor Name'] == 'Benedict Smith':
        data_raw['Actor Height'] = 1.78
    return data_raw

# Apply the function to each row
df_actor_height = df_actor_height.apply(change_height, axis=1)

# Rounding the heights
df_actor_height['Actor Height'] = round(df_actor_height['Actor Height'],1)  

# Loading of the edited ethnicity.csv file that had been enriched with the translation of the ethnicity ID, and other classification columns
df_ethnicities_translated = pd.read_csv('../Extra/ethnicity_translated.csv', sep=';')

# Replace all the Ethnicity ID by the translated ethnicities + adding the country of origins of the actors
df_character_ethnicities = pd.DataFrame(df_character_metadata_B['Actor Ethnicity'])

df_character_ethnicities = df_character_ethnicities.merge(df_ethnicities_translated, 
    left_on='Actor Ethnicity', right_on='Ethnicity ID', how='left', suffixes=('', '_translated'))

# Selecting relevant columns and dropping duplicates
df_character_ethnicities = df_character_ethnicities[['Ethnicity', 'Country of Origin']]


# Define a function to split the string
def split(string):
    if isinstance(string, str):  # Check if it is a string
        return string.split(', ')
    return string  # Return as is if not a string

# Apply the function to the  column
df_character_ethnicities['Ethnicity'] = df_character_ethnicities['Ethnicity'].apply(split)
df_character_ethnicities['Country of Origin'] = df_character_ethnicities['Country of Origin'].apply(split)

# Extract the 'Actor age at movie realise' column as a Series
df_actor_ages = df_character_metadata_B['Actor age at movie realise']

def negative_to_NaN(data_raw):
    '''Change all the negative values to NaN'''
    if data_raw < 0:
        data_raw = np.nan
    return data_raw

# Apply the function to the Series without using axis
df_actor_ages = df_actor_ages.apply(negative_to_NaN)

# Putting all the preprocessed features of interest together
df_character_metadata_C = pd.DataFrame({'Freebase Movie ID': df_character_metadata_B['Freebase Movie ID'],
                                        'Character Name': df_character_metadata_B['Character Name'],
                                        'Actor Name': df_character_metadata_B['Actor Name'],
    'Movie Release Date': df_charcter_movie_release_date, 'Actor Gender': df_character_metadata_B['Actor Gender'], 
    'Actor Height': df_actor_height['Actor Height'], 'Actor Ethnicity': df_character_ethnicities['Ethnicity'],
    'Actor Country of Origin': df_character_ethnicities['Country of Origin'], 'Actor Age': df_actor_ages })

# Removing the rows that were previously assigned to NaN
df_character_metadata_D = df_character_metadata_C#.dropna()

df_character_metadata_D.to_csv('src/data/characters_preprecessed_for_archetype.csv', index=False)