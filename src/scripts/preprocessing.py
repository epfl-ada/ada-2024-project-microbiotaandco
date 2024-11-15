import pandas as pd
import numpy as np
import ast
import re
import os

def change_height(data_raw):
    """
    Corrects the height of actors based on their names.
    
    This function adjusts the 'Actor Height' column in the input data 
    based on the actor's name. It handles specific cases for actors 
    where the height is known to be incorrect or missing in the raw data.

    Parameters:
        data_raw (pandas.DataFrame): A row of actor data containing 
                                      'Actor Name' and 'Actor Height' columns.
    
    Returns:
        pandas.DataFrame: The input row with corrected 'Actor Height' 
                          based on predefined actor names.
    """
    
    # Check if the actor's name matches a known case and adjust their height accordingly
    if data_raw['Actor Name'] == 'Zohren Weiss':
        # Set height to NaN if the actor's name is 'Zohren Weiss'
        data_raw['Actor Height'] = np.nan
    elif data_raw['Actor Name'] == 'Vince Corazza':
        # Set height to 1.78 meters for 'Vince Corazza'
        data_raw['Actor Height'] = 1.78
    elif data_raw['Actor Name'] == 'Benedict Smith':
        # Set height to 1.78 meters for 'Benedict Smith'
        data_raw['Actor Height'] = 1.78

    # Return the modified data row
    return data_raw


def split(string):
    """
    Splits a string into a list based on a comma and space separator.
    
    This function is useful for splitting a string that contains multiple 
    items separated by ', '. If the input is not a string, it returns the 
    input as is without modification.
    
    Parameters:
        string (str or other): The input to be split, typically a string of 
                                comma-separated values. If not a string, 
                                it is returned unchanged.
        
    Returns:
        list or original: If the input is a string, it returns a list of 
                          substrings split by ', '. Otherwise, returns 
                          the original input.
    """
    # Check if the input is a string
    if isinstance(string, str):
        # Split the string at each comma and space
        return string.split(', ')
    
    # If the input is not a string, return it as is
    return string


def negative_to_NaN(data_raw):
    """
    Converts all negative values in the input data to NaN.
    
    This function checks if the input value is negative. If it is, 
    the value is replaced with NaN. This is typically used to handle 
    invalid or erroneous negative values that should not be present.

    Parameters:
        data_raw (numeric or pandas.Series): A numeric value or a pandas Series to check for negative values.

    Returns:
        numeric or pandas.Series: The input value with any negative values replaced by NaN.
    """
    
    # Check if the input value is negative
    if data_raw < 0:
        # Replace negative value with NaN
        data_raw = np.nan

    # Return the modified value (or original if no change was needed)
    return data_raw


def create_dictionary(feature):
    """
    Creates a dictionary of language translations from a list of dictionaries.

    This function iterates over a list where each element is expected to be a dictionary, 
    with language codes as keys and translations as values. It constructs a dictionary 
    that contains unique language-translation pairs, ensuring that only the first translation 
    found for each language is stored.

    Parameters:
        feature (iterable): A list of dictionaries where each dictionary contains language codes 
                             as keys and their corresponding translations as values.

    Returns:
        dict: A dictionary where the keys are language codes and the values are their corresponding translations.
    """
    
    # Initialize an empty dictionary where each key will store a single translation (string)
    dictionary = {}

    # Iterate through each row in the feature list, where each element is expected to be a dictionary
    for d in feature:
        if isinstance(d, dict):  # Check if the element is a dictionary
            # Iterate through each key-value pair in the dictionary
            for lang, translation in d.items():
                # Add the language and translation to the dictionary if it's not already present
                # This ensures only the first translation found for each language is stored
                if lang not in dictionary:
                    dictionary[lang] = translation  # Store the translation directly as a string

    # Return the populated dictionary
    return dictionary


# Changing: [] --> NaN
def replace_empty_list(data_raw):
    """
    Replace empty lists with NaN.

    This function checks if the input value is an empty list and, if so, replaces it with NaN. 
    It returns the input value unchanged if it's not an empty list.

    Parameters:
        data_raw: The input value to be checked. It can be any type, but this function specifically 
                  targets empty lists.

    Returns:
        The input value, either unchanged or replaced with NaN if it is an empty list.
    """
    
    # Check if the input is an empty list
    if data_raw == []:
        # Replace empty list with NaN
        data_raw = np.nan
    
    # Return the modified or unchanged input
    return data_raw


# Function to find surrogate characters
def has_surrogates(text):
    """
    Check if the given text contains surrogate pairs.

    This function checks whether the input string contains any surrogate characters, 
    which are used to represent characters outside the Basic Multilingual Plane (BMP) 
    in UTF-16 encoding. Surrogate pairs are a combination of two 16-bit code units 
    used to encode characters that can't be represented in a single 16-bit unit.

    Parameters:
        text (str): The input text to check for surrogate pairs.

    Returns:
        bool: True if the input string contains surrogate characters, False otherwise.
    """
    
    # Check if the input is a string
    if isinstance(text, str):
        # Use regular expression to detect surrogate characters (U+D800 to U+DFFF range)
        return bool(re.search(r'[\ud800-\udfff]', text))
    
    # Return False if the input is not a string
    return False


def remove_surrogates(text):
    """
    Remove surrogate pairs from the given text.

    Surrogate pairs are used to represent characters outside the Basic Multilingual Plane 
    (BMP) in UTF-16 encoding. This function removes any characters that fall within the 
    surrogate pair range (U+D800 to U+DFFF).

    Parameters:
        text (str): The input string from which surrogate pairs should be removed.

    Returns:
        str: The input string with surrogate pairs removed. If the input is not a string, 
             it is returned unchanged.
    """
    
    # Check if the input is a string
    if isinstance(text, str):
        # Use regular expression to remove surrogate characters (U+D800 to U+DFFF range)
        return re.sub(r'[\ud800-\udfff]', '', text)
    
    # Return the input unchanged if it is not a string
    return text


def process_movie_dictionaries(df_column, output_csv_path, translation_dict=None):
    """
    Process a column of movie languages, handles surrogate characters, and applies translations.

    This function processes a column of movie languages represented as dictionaries or strings, 
    removes any surrogate characters, replaces language IDs with their translations, and saves 
    the translated dictionary as a CSV file. If no translation dictionary is provided, one is 
    created from the provided column data. The function also ensures that empty lists are replaced 
    with NaN values.

    Parameters:
        df_column (pd.Series): A pandas Series containing movie languages as dictionaries or strings.
                               Each dictionary maps language IDs to their corresponding translations.
        output_csv_path (str): The file path where the cleaned and translated dictionary will be saved as a CSV.
        translation_dict (dict, optional): A dictionary of language translations. If None, one will be created 
                                           using the provided column data.

    Returns:
        pd.Series: A cleaned and processed pandas Series where language IDs are replaced by their translations, 
                   surrogate characters are removed, and empty lists are replaced with NaN.
    """
    # Convert strings to dictionaries if the element is a string
    df_column = df_column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Create the translation dictionary if one is not provided
    if translation_dict is None:
        translation_dict = create_dictionary(df_column)

    # Convert the translation dictionary to a DataFrame and save to a CSV file
    df_translation_dict = pd.DataFrame(list(translation_dict.items()), columns=['Language', 'Translation'])
    df_translation_dict.to_csv(output_csv_path, index=False, encoding='utf-8', errors='replace')

    # Check for problematic entries (e.g., surrogate characters)
    for key, value in translation_dict.items():
        if has_surrogates(key) or has_surrogates(value):
            print(f"Problematic entry: {key} -> {value}")

    # Clean the translation dictionary by removing surrogate characters
    cleaned_translation_dict = {
        remove_surrogates(key): remove_surrogates(value)
        for key, value in translation_dict.items()
    }

    # Convert dictionary entries into lists of language keys
    df_column = df_column.apply(lambda x: list(x.keys()) if isinstance(x, dict) else x)
    
    # Replace movie language IDs in the list with their corresponding translations
    df_column = df_column.apply(
        lambda x: [cleaned_translation_dict.get(language, language) for language in x] if isinstance(x, list) else x
    )
    
    # Ensure that all language entries are split and expanded
    df_column = df_column.apply(split)
    
    # Replace empty lists with NaN values
    df_column = df_column.apply(replace_empty_list)

    return df_column

def preprocessing_datasets():
    """
    Preprocesses character and movie metadata, including height, ethnicity, age, and language.

    This function performs several preprocessing steps on two datasets: character metadata and movie metadata.
    It processes columns related to actor's height, ethnicity, age, and movie language, merges the data, and 
    outputs both a compact and exploded version of the resulting dataset. The function also saves intermediate
    files, such as ethnicity translations and language dictionaries.

    Steps performed:
    1. Preprocess actor height by applying custom corrections.
    2. Process actor ethnicity by merging with a translated dataset and splitting the ethnicities.
    3. Clean actor age by replacing negative values with NaN.
    4. Process movie metadata, including release date, box office revenue, and language.
    5. Merge character and movie metadata on 'Freebase Movie ID'.
    6. Create exploded versions of the merged dataset for specified columns (e.g., language, ethnicity).
    7. Save the processed data to CSV files.

    Returns:
        None: This function saves the processed data to CSV files and does not return any values.

    Notes:
        - Ethnicity translations are expected to be in a file `../data/ethnicity_translated.csv`.
        - The language dictionary is processed and saved to `../data/language_dict.csv`.
        - The function creates two output files:
          - `metadata_OI_compact.csv`: The compact version of the metadata.
          - `metadata_OI_exploded.csv`: The exploded version with lists expanded.

    """
    current_path = os.getcwd()
    print(current_path)
    
    # Load character and movie metadata
    df_character_metadata = pd.read_csv(current_path + "/src/data/character_metadata.csv")
    df_movie_metadata = pd.read_csv(current_path + "/src/data/movie_metadata.csv")

    ## Preprocessing of Character Metadata
    df_character_metadata_A = df_character_metadata[['Freebase Movie ID','Actor Name', 'Actor Gender', 'Actor Height', 'Actor Ethnicity', 'Actor age at movie realise']]
   
    print("Creation of the compact dataset...")
    ## Actor Height
    # Create a new DataFrame with specific columns
    df_actor_height = df_character_metadata_A[['Actor Name', 'Actor Height']].copy()

    # Apply the function to each row
    df_actor_height = df_actor_height.apply(change_height, axis=1)

    # Rounding the heights
    df_actor_height['Actor Height'] = round(df_actor_height['Actor Height'],1)  

    ## Actor Ethnicity
    # Create a file with all the ethnicity IDs for translation
    df_ethnicities = df_character_metadata_A['Actor Ethnicity'].value_counts().reset_index()
    df_ethnicities.to_csv(current_path + '/src/data/ethnicity.csv')

    # Load the edited ethnicity file with translations and other classification columns
    df_ethnicities_translated = pd.read_csv( current_path + '/src/data/ethnicity_translated.csv', sep=';')

    # Replace ethnicity IDs with translated ethnicities and add country of origin
    df_character_ethnicities = pd.DataFrame(df_character_metadata_A['Actor Ethnicity'])
    df_character_ethnicities = df_character_ethnicities.merge(df_ethnicities_translated, 
        left_on='Actor Ethnicity', right_on='Ethnicity ID', how='left', suffixes=('', '_translated'))

    # Select relevant columns and drop duplicates
    df_character_ethnicities = df_character_ethnicities[['Ethnicity', 'Country of Origin']]

    # Ensure that all ethnicities are taken into account
    df_character_ethnicities['Ethnicity'] = df_character_ethnicities['Ethnicity'].apply(split)
    df_character_ethnicities['Country of Origin'] = df_character_ethnicities['Country of Origin'].apply(split)

    ## Age
    # Process the 'Actor age at movie realise' column
    df_actor_ages = df_character_metadata_A['Actor age at movie realise']
    df_actor_ages = df_actor_ages.apply(negative_to_NaN)

    # Combine the preprocessed character data features into a single DataFrame
    df_character_metadata_B = pd.DataFrame({
        'Freebase Movie ID': df_character_metadata_A['Freebase Movie ID'], 
        'Actor Gender': df_character_metadata_A['Actor Gender'], 
        'Actor Height': df_actor_height['Actor Height'], 
        'Actor Ethnicity': df_character_ethnicities['Ethnicity'],
        'Actor Country of Origin': df_character_ethnicities['Country of Origin'], 
        'Actor Age': df_actor_ages 
    })
    
    ## Movie Metadata
    # Keep only the relevant features
    df_movie_metadata_A = df_movie_metadata[['Freebase Movie ID', 'Movie Release Date', 'Movie Box Office Revenue', 'Movie Language', 'Movie Country']]

    # Process Movie Release Date
    df_movie_release_date = df_movie_metadata_A['Movie Release Date'].str[:4]
    df_movie_release_date = pd.to_numeric(df_movie_release_date, errors='coerce')

    # Process Movie Language
    df_movie_language_processed = process_movie_dictionaries(df_movie_metadata_A['Movie Language'], output_csv_path= current_path + '/src/data/language_dict.csv')
    df_movie_country_processed = process_movie_dictionaries(df_movie_metadata_A['Movie Country'], output_csv_path=current_path +'/src/data/country_dict.csv')

    # Combine the preprocessed movie data into a single DataFrame
    df_movie_metadata_B = pd.DataFrame({
        'Freebase Movie ID': df_movie_metadata_A['Freebase Movie ID'],
        'Movie Release Date': df_movie_release_date, 
        'Movie Box Office Revenue': df_movie_metadata_A['Movie Box Office Revenue'],
        'Movie Language': df_movie_language_processed, 
        'Movie Country': df_movie_country_processed
    })

    ## Merge DataFrames on 'Freebase Movie ID'
    df_metadata_OI_compact = pd.merge(df_character_metadata_B, df_movie_metadata_B[['Freebase Movie ID','Movie Release Date', 'Movie Box Office Revenue', 'Movie Language', 'Movie Country']],
                            on='Freebase Movie ID', how='inner')

    # Display the resulting DataFrame and save to CSV
    df_metadata_OI_compact = df_metadata_OI_compact[['Freebase Movie ID', 'Movie Release Date', 'Movie Box Office Revenue', 'Movie Language', 'Movie Country', 'Actor Gender','Actor Height', 'Actor Age', 'Actor Ethnicity', 'Actor Country of Origin']]

    # Removing NAs
    df_metadata_OI_compact = df_metadata_OI_compact.dropna()

    ## Creation of the exploded dataset
    df_metadata_OI_exploded = df_metadata_OI_compact
    columns_to_explode = ['Movie Language', 'Movie Country', 'Actor Ethnicity', 'Actor Country of Origin']

    # Exploding each column in the list
    for column in columns_to_explode:
        df_metadata_OI_exploded = df_metadata_OI_exploded.explode(column)

    # Resetting the index after all explosions
    df_metadata_OI_exploded = df_metadata_OI_exploded.reset_index(drop=True)

    # Save the preprocessed and exploded datasets
    df_metadata_OI_compact.to_csv(current_path + '/data/metadata_OI_compact.csv', index=False)
    print("done.")

    print("Creation of the exploded dataset...")
    df_metadata_OI_exploded.to_csv(current_path + '/data/metadata_OI_exploded.csv', index=False)
    print("done.")
