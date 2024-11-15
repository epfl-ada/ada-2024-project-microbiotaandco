import matplotlib.pyplot as plt


df_merged = df_result_age \
    .merge(df_result_height, on='Freebase Movie ID') \
    .merge(df_result_ethnicity, on='Freebase Movie ID') \
    .merge(df_result_gender, on='Freebase Movie ID')\
    .merge(df_result_foreigners, on='Freebase Movie ID')

df_merged['diversity_score'] = df_merged[['age_score', 'height_score', 'ethnicity_score', 'gender_score','Foreign Actor Proportion']].mean(axis=1)
df_merged.head(10)

#adding IDs associated with the movie metadata
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