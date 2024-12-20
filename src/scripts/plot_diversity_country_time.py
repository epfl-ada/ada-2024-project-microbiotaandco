

from scipy.stats import chi2_contingency
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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








################## LINEAR REGRESSION LINES



#List of diversity scores and their respective labels for plotting
diversity_scores = ['gender_score', 'age_score', 'height_score', 'ethnicity_score', 'Foreign Actor Proportion', 'diversity_score']
score_labels = ['Gender', 'Age', 'Height', 'Ethnicity', 'Foreign Actors', 'Diversity Score']
colors = ['thistle', 'palegoldenrod', 'lightsalmon', 'mediumseagreen', 'lightblue', 'indianred'] 

#Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X_centered = df_yearly['Movie Release Date']
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))

    # Plot the original data and regression line
    plt.plot(df_yearly['Movie Release Date'], df_yearly[score],  color=colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    plt.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Linear Regression line')
    plt.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color=colors[i], alpha=0.5, label='Mean Score Area')

    # Add R-squared and regression 
    r_squared = model.score(X, y)
    equation_text = f'{score_labels[i]} Score = {model.coef_[0]:.4f} * Year + {model.intercept_:.2f}'
    r_squared_text = f'R² = {r_squared:.4f}'

    plt.figtext(0.03, -0.02, equation_text, ha='left', va='top', fontsize=16, color='red')
    plt.figtext(0.03, -0.08, r_squared_text, ha='left', va='top', fontsize=16, color='black')

    plt.xlabel('Movie release year', fontsize=14)
    plt.ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    plt.title(f'Time evolution of the mean {score_labels[i]} score.', fontsize=14)
    plt.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()



################## PLOT THE REGRESSION COEFFICIENTS



# Calculate mean or median for each diversity metric
df_yearly_gender = Diversity_movie_metadata.groupby('Movie Release Date')['gender_score'].mean().reset_index()
df_yearly_ethnicity = Diversity_movie_metadata.groupby('Movie Release Date')['ethnicity_score'].mean().reset_index()
df_yearly_age = Diversity_movie_metadata.groupby('Movie Release Date')['age_score'].mean().reset_index()
df_yearly_foreign_actor = Diversity_movie_metadata.groupby('Movie Release Date')['Foreign Actor Proportion'].mean().reset_index()
df_yearly_diversitymean = Diversity_movie_metadata.groupby('Movie Release Date')['diversity_score'].mean().reset_index()
df_yearly_height = Diversity_movie_metadata.groupby('Movie Release Date')['height_score'].median().reset_index()


# Make a list of dataframes
dfs = [
    ('Gender Score', df_yearly_gender), 
    ('Ethnicity Score', df_yearly_ethnicity), 
    ('Age Score', df_yearly_age), 
    ('Foreign Actor Proportion', df_yearly_foreign_actor), 
    ('Height Score', df_yearly_height),
    ('Diversity Score', df_yearly_diversitymean) 
    
]

# Perform linear regression over every dataframe in the list and store the results
results = []
for label, df in dfs:
    X = df['Movie Release Date'].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    results.append((label, model.coef_[0], model.intercept_))

# Create the bar plot
plt.figure(figsize=(10, 6))

# Make a barplot of the coefficients
labels = [result[0] for result in results]
coefficients = [result[1] for result in results]

bars = plt.bar(labels, coefficients, color='darkgrey', alpha=0.7)

# Add the exact value of the regression slope for each coefficient above each bar
for bar, (label, coef, _) in zip(bars, results):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() , 
             f'{coef:.4f}', ha='center', va='bottom', fontsize=12, color='black')

# Labels and title for the plot
plt.xlabel('Diversity type', fontsize=14)
plt.ylabel('Regression coefficient (slope)', fontsize=14)
plt.title('Linear regression coefficients for different diversity metrics.', fontsize=16)

# Format x-axis labels
plt.xticks(rotation=45, ha='right')

# Adding grid for better visibility of values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



##################CHOOSING THE DEGREE OF THE POLYNOMIAL REGRESSION
# Range of polynomial degrees to test
degrees = range(1, 15)

# Store results for plotting
results = {score: {'degrees': [], 'r_squared': []} for score in diversity_scores}

# Loop over each diversity score
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    df_yearly = data_copy.groupby('Movie Release Date')[score].mean().reset_index()

    X_centered = df_yearly['Movie Release Date'].values.reshape(-1, 1)
    y = df_yearly[score].values

    # Test each polynomial degree
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_centered)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        r_squared = model.score(X_poly, y)

        # Store results
        results[score]['degrees'].append(degree)
        results[score]['r_squared'].append(r_squared)

# Plot R² for all scores on a single graph
plt.figure(figsize=(12, 8))

# Different colors for each score
colors = ['thistle', 'palegoldenrod', 'lightsalmon' ,'mediumseagreen', 'lightblue' , 'indianred'  ]  
for i, score in enumerate(diversity_scores):
    plt.plot(
        results[score]['degrees'],
        results[score]['r_squared'],
        marker='o',
        label=f'{score_labels[i]} R²',
        color=colors[i],
        linestyle='-'
    )

# Add labels, title, and legend
plt.xlabel('Polynomial Degree', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Polynomial model performance for different degrees and diversity scores.', fontsize=16)
plt.grid(True)
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()

# Print the best degree based on R² and MSE for each score
for i, score in enumerate(diversity_scores):
    best_degree_r2 = results[score]['degrees'][np.argmax(results[score]['r_squared'])]
    print(f'{score_labels[i]}:')
    print(f'  Best degree based on R²: {best_degree_r2}')



################## POLYNOMIAL REGRESSION LINES WITH CRITICAL POINTS AND INFLECTION POINTS


# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X_centered = df_yearly['Movie Release Date']
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  

    # Polynomial feature creation with defined degree
    degree = 3
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    coeffs = model.coef_
    intercept = model.intercept_

    # Create the polynomial function
    poly_coeff = np.append(intercept, coeffs[1:])
    p = np.poly1d(poly_coeff[::-1])  # Reverse the coefficients for np.poly1d

    # Derivatives for critical and inflection points
    p_prime = p.deriv()  # First derivative
    p_double_prime = p_prime.deriv()  # Second derivative

    # Find critical and inflection points
    critical_points = p_prime.r
    inflection_points = p_double_prime.r

    # Create a new figure for each diversity score
    plt.figure(figsize=(10, 6))

    # Plot the original data and regression line
    plt.plot(df_yearly['Movie Release Date'], df_yearly[score],  color= colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    plt.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Polynomial Regression line')
    plt.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], alpha=0.5, label='Mean Score Area')

    # Scatter the critical and inflection points
    plt.scatter(critical_points, p(critical_points), color='darkred', s=100, zorder=5, label='Critical points')
    plt.scatter(inflection_points, p(inflection_points), color='steelblue', s=100, zorder=5, label='Inflection points')

    # Add R-squared and regression equation text below the plot
    r_squared = model.score(X_poly, y)
    equation_text = f"y = {intercept:.3f} + " + " + ".join([f"{coeffs[i]:.3f}x^{i}" for i in range(1, len(coeffs))])
    r_squared_text = f'R² = {r_squared:.4f}'

    # Place the regression equation and R-squared value outside the plot
    plt.figtext(0.05, -0.05, equation_text, ha='left', va='top', fontsize=12, color='red')
    plt.figtext(0.05, -0.08, r_squared_text, ha='left', va='top', fontsize=12, color='black')
    plt.figtext(0.05 , -0.02, f'Polynomial degree = {degree}', ha='left', va='top', fontsize=12, color='black')

    # Set labels and title for the plot
    plt.xlabel('Movie release year', fontsize=14)
    plt.ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    plt.title(f'Time evolution of the mean {score_labels[i]} score.', fontsize=14)
    plt.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')
    plt.legend()

    print(f'Polynomial regression coefficients for {score_labels[i]}: {coeffs}')
    print(f'Polynomial regression intercept for {score_labels[i]}: {intercept}')

    # Adjust layout to avoid overlap and add extra space for equation and R-squared
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()







##################ANALYSIS OF THE INTERESTING ERA

movie_historical_analysis_data = df_metadata_OI_exploded.copy()
movie_historical_analysis_data.head()


#extract all the rows with films ranging from 1950 to 1970
movie_md_50_70 = movie_historical_analysis_data[(movie_historical_analysis_data['Movie Release Date'] >= 1950) & (movie_historical_analysis_data['Movie Release Date'] <= 1970)]
movie_md_pre50 = movie_historical_analysis_data[(movie_historical_analysis_data['Movie Release Date'] < 1950)]



# Count the number of movies for each ethnicity group
ethnicity_counts = movie_md_50_70['Actor Ethnicity'].value_counts().reset_index(name='Count')
ethnicity_counts.columns = ['Ethnicity', 'Count']

# Select top 5 ethnicities with the most movies
top_ethnicities = ethnicity_counts.head(5)['Ethnicity'].values

# Filter the data to include only top 5 ethnicities
movie_md_50_70_top = movie_md_50_70[movie_md_50_70['Actor Ethnicity'].isin(top_ethnicities)]
movie_md_pre50_top = movie_md_pre50[movie_md_pre50['Actor Ethnicity'].isin(top_ethnicities)]

# Group by release date and ethnicity, counting the number of movies
movie_counts_50_70_top = movie_md_50_70_top.groupby(['Movie Release Date', 'Actor Ethnicity']).size().reset_index(name='Movie Count')
movie_counts_pre50_top = movie_md_pre50_top.groupby(['Movie Release Date', 'Actor Ethnicity']).size().reset_index(name='Movie Count')

# Concatenate datasets for both periods
movie_counts_top = pd.concat([movie_counts_50_70_top, movie_counts_pre50_top], axis=0)

# Create a line plot for top ethnicities
g = sns.relplot(
    data=movie_counts_top,
    x="Movie Release Date", y="Movie Count", hue="Actor Ethnicity", kind="line", 
    style="Actor Ethnicity", markers=True, height=6, aspect=1.5
)
g._legend.remove()
# Customize the plot with titles and labels
plt.title('Number of Movies Released by Top Ethnicities (1950-1970 and Pre-1950)', fontsize=16)
plt.xlabel('Movie Release Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
