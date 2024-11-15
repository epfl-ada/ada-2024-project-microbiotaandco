import numpy as np
import pandas as pd

def calculate_gender_diversity(df):
    #This function computes gender diversity in a list of aactors
    #inputs: df with column actor genders as 'M' or 'F'
    #outputs: scalar value  gender diversity
    count_male = (df['Actor Gender'] == 'M').sum()
    count_female = (df['Actor Gender'] == 'F').sum()
    total_count = count_male+count_female
    proportion_m = count_male / total_count
    proportion_f = count_female / total_count
    #calculating how balanced the proportions of males and females are
    return 1 - np.abs(proportion_m - proportion_f)


def calculate_ethnicity_diversity(df):
   
    #This Function to calculate ethnicity diversity using Simpson's diversity index,
    #  which is a common measure for quantifying diversity.
    #inputs: df with column Actor Ethnicity as 'string'
    #outputs: scalar value  gender diversity

    ethnicity_counts = df['Actor Ethnicity'].value_counts()
    total_count = len(df)
    proportions = ethnicity_counts / total_count
    ethnicity_diversity = 1 - sum(proportions ** 2)
    return ethnicity_diversity


def calculate_age_diversity(df):
    #This Function to calculate age  diversity using Simpson's diversity index,
    #inputs: df with column Actor Age as int
    #outputs: scalar value age diversity

    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    df['Age Range'] = pd.cut(df['Actor Age'], bins=age_bins, labels=age_labels, right=False)
    age_counts = df['Age Range'].value_counts()
    total_count = len(df)
    proportions = age_counts / total_count
    simpson_diversity = 1 - sum(proportions ** 2)
    num_age_ranges = len(age_counts)
    if num_age_ranges > 1:
        max_simpson_diversity = 1 - (1 / num_age_ranges) ** 2
        age_diversity = simpson_diversity / max_simpson_diversity
    else:
        age_diversity = 0
    
    return age_diversity

def calculate_height_diversity(df):
    #This Function to calculate height diversity using Simpson's diversity index,
    #inputs: df with column Actor Height as double
    #outputs: scalar value height diversity
    height_counts = df['Actor Height'].value_counts()
    total_count = len(df)

    proportions = height_counts / total_count
    
    simpson_diversity = 1 - sum(proportions ** 2)
    
    num_unique_heights = len(height_counts)

    if num_unique_heights > 1:
        max_simpson_diversity = 1 - (1 / num_unique_heights) ** 2 
        height_diversity = simpson_diversity / max_simpson_diversity
    else:
        height_diversity = 0
    
    return height_diversity

def calculate_foreign_actor_proportion(df):
    #This Function to calculate foreign actor proportion
    #inputs: df with column Movie Country as List, 'Actor Country of Origin' as string, Freebase Movie Id as string
    #outputs: scalar value foreign actor proportion
    proportions = []
    for movie_id, group in df.groupby('Freebase Movie ID'):
        movie_countries = set(group['Movie Country'].iloc[0])  # Get unique movie country list
        actor_countries = group['Actor Country of Origin']  # List of actor countries
        foreign_actors = sum(1 for country in actor_countries if country not in movie_countries)
        total_actors = len(actor_countries)
        proportion = foreign_actors / total_actors if total_actors > 0 else 0
        proportions.append({
            'Freebase Movie ID': movie_id,
            'Foreign Actor Proportion': proportion
        })
    result_df = pd.DataFrame(proportions)
    return result_df


movie_info = df_metadata_OI_exploded.groupby('Freebase Movie ID').agg({
    'Movie Country': lambda x: list(x.unique()),  
    'Movie Language': lambda x: list(x.unique()), 
    'Movie Release Date': 'first',                
    'Movie Box Office Revenue': 'first'           
}).reset_index()


df_merged_1 = df_metadata_OI_exploded.drop(columns=['Movie Country', 'Movie Language', 'Movie Release Date', 'Movie Box Office Revenue']) \
                          .merge(movie_info, on='Freebase Movie ID', how='left')


df_merged_1 = df_merged_1[[
    'Freebase Movie ID', 'Movie Country', 'Movie Language', 'Movie Release Date', 'Movie Box Office Revenue',
    'Actor Age', 'Actor Gender', 'Actor Ethnicity', 'Actor Height', 'Actor Country of Origin'
]]


df_merged_unique = df_merged_1.drop_duplicates(subset=[
    'Freebase Movie ID', 'Actor Height', 'Actor Ethnicity', 'Actor Age', 'Actor Gender', 'Actor Country of Origin'
], keep='first')

count = df_merged_unique['Freebase Movie ID'].value_counts().get('/m/011yfd', 0)

df_metadata_OI = df_merged_unique[df_merged_unique['Freebase Movie ID'].map(df_merged_unique['Freebase Movie ID'].value_counts()) > 1]
df_metadata_OI.head(100)