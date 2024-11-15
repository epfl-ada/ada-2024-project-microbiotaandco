import pandas as pd
import json 


def formating_datasets():
    """
    Reads, processes, and formats several raw datasets related to movies and characters.
    The processed datasets are saved as CSV files for easier access and analysis.
    
    The datasets processed are:
    - Character Metadata
    - Movie Metadata
    - Name Clusters
    - Plot Summaries
    - TV Tropes Clusters
    
    Each dataset is read from its raw form, appropriate column names are added,
    and any additional necessary transformations are applied. Finally, the datasets
    are saved as structured CSV files.
    
    Outputs:
    - 'character_metadata.csv'
    - 'movie_metadata.csv'
    - 'name_clusters.csv'
    - 'plot_summaries.csv'
    - 'tvtropes_clusters.csv'
    """
    # Load raw datasets
    character_metadata = pd.read_csv("../data/character.metadata.tsv", sep='\t', header=None)
    movie_metadata = pd.read_csv("../data/movie.metadata.tsv", sep="\t", header=None)
    name_clusters = pd.read_csv("../data/name.clusters.txt", sep="\t", header=None)
    plot_summaries = pd.read_csv("../data/plot_summaries.txt", sep="\t", header=None)
    tvtropes_clusters = pd.read_csv("../data/tvtropes.clusters.txt", sep="\t", header=None)

    # --- Character Metadata ---
    # Add column names for character metadata
    character_metadata.columns = [
        "Wikipedia Movie ID", "Freebase Movie ID", "Movie Release Date",
        "Character Name", "Actor date of birth", "Actor Gender", "Actor Height",
        "Actor Ethnicity", "Actor Name", "Actor age at movie realise",
        "Freebase Character Map ID", "Freebase character ID", "Freebase actor ID"
    ]
    # Save formatted dataset
    character_metadata.to_csv('../data/character_metadata.csv', index=False)

    # --- Movie Metadata ---
    # Add column names for movie metadata
    movie_metadata.columns = [
        "Wikipedia Movie ID", "Freebase Movie ID", "Movie Name", "Movie Release Date",
        "Movie Box Office Revenue", "Movie Runtime", "Movie Language",
        "Movie Country", "Movie Genre"
    ]
    # Save formatted dataset
    movie_metadata.to_csv('../data/movie_metadata.csv', index=False)

    # --- Name Clusters ---
    # Add column names for name clusters
    name_clusters.columns = ["Character Name", "Freebase Character Map ID"]
    # Save formatted dataset
    name_clusters.to_csv('../data/name_clusters.csv', index=False)

    # --- Plot Summaries ---
    # Add column names for plot summaries
    plot_summaries.columns = ["Wikipedia Movie ID", "Summary"]
    # Save formatted dataset
    plot_summaries.to_csv('../data/plot_summaries.csv', index=False)

    # --- TV Tropes Clusters ---
    # Add column names for TV tropes clusters
    tvtropes_clusters.columns = ["Character Types", "Freebase Character Map ID"]

    # Initialize list to hold processed Freebase Character Map IDs
    list_inside_FCM_ID = []

    for el in tvtropes_clusters["Freebase Character Map ID"]:
        # Parse the Freebase Character Map ID, which is stored as a JSON string
        parsed_el = json.loads(el)
        # Convert the parsed JSON into a DataFrame
        new_data = pd.DataFrame([parsed_el])
        # Append the DataFrame to the list
        list_inside_FCM_ID.append(new_data)

    # Combine all parsed Freebase Character Map ID DataFrames into one
    inside_FCM_ID = pd.concat(list_inside_FCM_ID).reset_index(drop=True)

    # Merge the parsed IDs with the original dataset
    tvtropes_clusters = pd.concat([tvtropes_clusters, inside_FCM_ID], axis=1)
    # Drop the original JSON column and rename the parsed ID column
    tvtropes_clusters = tvtropes_clusters.drop(columns="Freebase Character Map ID")
    tvtropes_clusters = tvtropes_clusters.rename(columns={"id": "Freebase Character Map ID"})

    # Save the formatted TV tropes clusters dataset
    tvtropes_clusters.to_csv('../data/tvtropes_clusters.csv', index=False)