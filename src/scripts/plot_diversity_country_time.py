

from scipy.stats import chi2_contingency
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import os

df_metadata_OI_exploded=pd.read_csv('data/metadata_OI_exploded.csv')


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





df_result_gender = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_gender_diversity).reset_index()
df_result_gender.columns = ['Freebase Movie ID', 'gender_score']
df_result_gender.head(10)



df_result_foreigners = calculate_foreign_actor_proportion(df_metadata_OI)
df_result_foreigners.head(10)



df_result_age = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_age_diversity).reset_index()
df_result_age.columns = ['Freebase Movie ID', 'age_score']



df_result_height = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_height_diversity).reset_index()
df_result_height.columns = ['Freebase Movie ID', 'height_score']


df_result_ethnicity = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_ethnicity_diversity).reset_index()
df_result_ethnicity.columns = ['Freebase Movie ID', 'ethnicity_score']



df_merged = df_result_age \
    .merge(df_result_height, on='Freebase Movie ID') \
    .merge(df_result_ethnicity, on='Freebase Movie ID') \
    .merge(df_result_gender, on='Freebase Movie ID')\
    .merge(df_result_foreigners, on='Freebase Movie ID')

df_merged['diversity_score'] = df_merged[['age_score', 'height_score', 'ethnicity_score', 'gender_score','Foreign Actor Proportion']].mean(axis=1)
df_merged.head(10)

Diversity_movie_metadata=df_merged.merge(
    df_merged_unique[['Freebase Movie ID', 'Movie Release Date', 'Movie Box Office Revenue', 'Movie Language', 'Movie Country']],
    on='Freebase Movie ID',
    how='inner') 
Diversity_movie_metadata.sample(10)






































df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['gender_score'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['gender_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('Movie Release Year', fontsize=14)
plt.ylabel('Mean Gender Score', fontsize=14)
plt.title('Mean Gender Score by Year', fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.show()

df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['ethnicity_score'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['ethnicity_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('Movie Release Year', fontsize=14)
plt.ylabel('Mean ethnicity Score', fontsize=14)
plt.title('Mean ethnicity Score by Year', fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.show()


df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['age_score'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['age_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('Movie Release Year', fontsize=14)
plt.ylabel('Mean Age Score', fontsize=14)
plt.title('Mean Age Score by Year', fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.show()

df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['height_score'].median().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['height_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('Movie Release Year', fontsize=14)
plt.ylabel('Mean height Score', fontsize=14)
plt.title('Mean height Score by Year', fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.show()


df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['Foreign Actor Proportion'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['Foreign Actor Proportion'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('Movie Release Year', fontsize=14)
plt.ylabel('Mean foreign actor Score', fontsize=14)
plt.title('Mean foreign Score by Year', fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.show()

df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['diversity_score'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['diversity_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('Movie Release Year', fontsize=14)
plt.ylabel('Mean diversity Score', fontsize=14)
plt.title('Mean diversity Score by Year', fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.show()


df_exploded = Diversity_movie_metadata.explode('Movie Country')
top_30_countries = df_exploded['Movie Country'].value_counts().head(30).index
df_top_countries = df_exploded[df_exploded['Movie Country'].isin(top_30_countries)]
country_gender_scores = df_top_countries.groupby('Movie Country')['gender_score'].mean()
plt.figure(figsize=(14, 8))
country_gender_scores = country_gender_scores.sort_values(ascending=False)
plt.bar(country_gender_scores.index, country_gender_scores.values, color='skyblue', width=0.6)
plt.title('Average Gender Score by Country (Top 30 Most Common)')
plt.xlabel('Country')
plt.ylabel('Gender Score')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

plt.show()


df_exploded = Diversity_movie_metadata.explode('Movie Country')
top_30_countries = df_exploded['Movie Country'].value_counts().head(30).index
df_top_countries = df_exploded[df_exploded['Movie Country'].isin(top_30_countries)]
country_ethnicity_scores = df_top_countries.groupby('Movie Country')['ethnicity_score'].mean()
plt.figure(figsize=(14, 8))
country_ethnicity_scores = country_ethnicity_scores.sort_values(ascending=False)
plt.bar(country_ethnicity_scores.index, country_ethnicity_scores.values, color='skyblue', width=0.6)
plt.title('Average ethnicity Score by Country (Top 30 Most Common)')
plt.xlabel('Country')
plt.ylabel('ethnicity Score')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

plt.show()



df_exploded = Diversity_movie_metadata.explode('Movie Country')
top_30_countries = df_exploded['Movie Country'].value_counts().head(30).index
df_top_countries = df_exploded[df_exploded['Movie Country'].isin(top_30_countries)]
country_age_scores = df_top_countries.groupby('Movie Country')['age_score'].mean()
plt.figure(figsize=(14, 8))
country_age_scores = country_age_scores.sort_values(ascending=False)
plt.bar(country_age_scores.index, country_age_scores.values, color='skyblue', width=0.6)
plt.title('Average age Score by Country (Top 30 Most Common)')
plt.xlabel('Country')
plt.ylabel('age Score')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

plt.show()

df_exploded = Diversity_movie_metadata.explode('Movie Country')
top_30_countries = df_exploded['Movie Country'].value_counts().head(30).index
df_top_countries = df_exploded[df_exploded['Movie Country'].isin(top_30_countries)]
country_height_scores = df_top_countries.groupby('Movie Country')['height_score'].mean()
plt.figure(figsize=(14, 8))
country_height_scores = country_height_scores.sort_values(ascending=False)
plt.bar(country_height_scores.index, country_height_scores.values, color='skyblue', width=0.6)
plt.title('Average height Score by Country (Top 30 Most Common)')
plt.xlabel('Country')
plt.ylabel('height Score')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

plt.show()


df_exploded = Diversity_movie_metadata.explode('Movie Country')
top_30_countries = df_exploded['Movie Country'].value_counts().head(30).index
df_top_countries = df_exploded[df_exploded['Movie Country'].isin(top_30_countries)]
country_foreign_scores = df_top_countries.groupby('Movie Country')['Foreign Actor Proportion'].mean()
plt.figure(figsize=(14, 8))
country_foreign_scores = country_foreign_scores.sort_values(ascending=False)
plt.bar(country_foreign_scores.index, country_foreign_scores.values, color='skyblue', width=0.6)
plt.title('Average Foreign actor proportion by Country (Top 30 Most Common)')
plt.xlabel('Country')
plt.ylabel('Foreign actor proportion')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

plt.show()