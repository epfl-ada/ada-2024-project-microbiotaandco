import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
    # Load the character metadata tsv file 
PATH_character_metadata="../data/character.metadata.tsv"
df_character_metadata = pd.read_table(PATH_character_metadata, sep="	", names=['Wiki_movie_ID', 'Freebase_movie_ID','Movie_release_date','Character_name','Actor_date_of_birth','Actor_gender','Actor_height','Actor_ethnicity_Freebase_ID','Actor_name','Actor_age','Freebase_character/actor_ID','Freebase_character_ID','Freebase_actor_ID'])
df_character_metadata=df_character_metadata[df_character_metadata['Actor_age']>-1]

    # Drop all rows such that the row/character has no Freebase character ID
df_character_metadata_clean=df_character_metadata.dropna(subset=['Freebase_character_ID'])

    # Create a list of data chunks (of lenght N), IMPORTANT this will permit to send for each query N Freebase character ID
N=100
list_df = [df_character_metadata_clean.iloc[i:i + N] for i in range(0, len(df_character_metadata_clean), N)]
print(len(list_df))

# Generate the SPARQL query and send it, then return a dictionnary with the results received from wikidata
def RequestToDictionnary(freebase_ids):
    query = f"""
        SELECT ?item ?itemLabel ?freebase_id (GROUP_CONCAT(?occupation; separator=", ") AS ?occupations) (GROUP_CONCAT(?occupationLabel; separator=", ") AS ?occupationLabels) WHERE {{
        VALUES ?freebase_id {{ {" ".join(f'"{id}"' for id in freebase_ids)} }}
        ?item wdt:P646 ?freebase_id.  # Matches the Freebase ID to the item
        OPTIONAL {{ 
            ?item wdt:P106 ?occupation.  # Occupation relation
            ?occupation rdfs:label ?occupationLabel.  # Label for occupations
            FILTER(LANG(?occupationLabel) = "en")  # Ensure English labels
        }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        GROUP BY ?item ?itemLabel ?freebase_id
        """

    # Define the URL for the Wikidata SPARQL endpoint
    url = "https://query.wikidata.org/sparql"

    # Send the request to the Wikidata SPARQL endpoint
    response = requests.get(url, params={'query': query, 'format': 'json'})

# Check for a successful response
    if response.status_code == 200:
        data = response.json()
        
        # Parse the results into a dictionary where Freebase ID (not Wikidata ID) is the key
        occupation_data = {}

        for result in data['results']['bindings']:
            freebase_id = result.get('freebase_id', {}).get('value', 'N/A')
            occupations = result.get('occupations', {}).get('value', 'N/A')  # Occupations as comma-separated string
            occupationLabels = result.get('occupationLabels', {}).get('value', 'N/A')  # Occupation labels as comma-separated string
            occupation_data[freebase_id] = {
                "Occupations": occupations,
                "Occupation Labels": occupationLabels
            }
        return occupation_data

    else:
        print(f"Error {response.status_code}: {response.reason}")


def get_occupation_data(freebase_id,occupation_data):
    return occupation_data.get(freebase_id, {"Occupations": 'N/A', "Occupation Labels": 'N/A'})

    # Loop through the data chunks, send queries for each data chunk and combine the result obtained from wikidata
Tt_dict={}
for df in list_df:
    freebase_ids=df["Freebase_character_ID"].tolist()
    occupation_data=RequestToDictionnary(freebase_ids)
    Tt_dict.update(occupation_data)
    time.sleep(2)            # IMPORTANT ELSE WE GET ERROR MESSAGE BECAUSE WE SEND TOO MANY REQUEST IN A GIVEN TIME SPAN

df_character_metadata_clean[['Occupations', 'Occupation Labels']] = df_character_metadata_clean['Freebase_character_ID'].apply(
        lambda x: pd.Series(get_occupation_data(x,Tt_dict)))

df_final=df_character_metadata_clean.replace('N/A', pd.NA).dropna(subset=['Occupations','Occupation Labels'])
df_final.to_csv('Extracted_Data.csv',index=False)

