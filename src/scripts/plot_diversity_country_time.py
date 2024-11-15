

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



#select the relevant columns , here excluded 'diversity_score' since it is an average
df_corr = df_merged[['age_score', 'height_score', 'ethnicity_score', 'gender_score', 'Foreign Actor Proportion']]

# correlation matrix
corr_matrix = df_corr.corr()

# heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', cbar=True)

plt.title('Correlation Heatmap of diversity scores', fontsize=16)
plt.tight_layout()
plt.show()



from sklearn.linear_model import LinearRegression


#mean gender score
df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['gender_score'].mean().reset_index()

#define variables
X = df_yearly['Movie Release Date'].values.reshape(-1, 1)  
y = df_yearly['gender_score'].values  

#linear regression model and fit 
model = LinearRegression()
model.fit(X, y)

#predict the gender score values using the model
y_pred = model.predict(X)

# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['gender_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5, label='Mean gender score')

#plot reg line
plt.plot(df_yearly['Movie Release Date'], y_pred, color='r', linestyle='--', linewidth=2, label='Linear Regression line')

#add regression equation 
equation_text = f'Gender Score = {model.coef_[0]:.4f} * Year + {model.intercept_:.2f}'
plt.text( df_yearly['Movie Release Date'].min(), max(y) * 0.95, equation_text, fontsize=8, color='darkred')

# Add labels and title
plt.xlabel('Movie release year', fontsize=14)
plt.ylabel('Mean gender score', fontsize=14)
plt.title('Mean gender score by year with LR fit', fontsize=16)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#show explicitly the regression model's coefficients and intercept
print(f'Linear regression coefficient: {model.coef_[0]}')
print(f'Linear regression intercept: {model.intercept_}')


#mean gender score
df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['ethnicity_score'].mean().reset_index()

#define variables
X = df_yearly['Movie Release Date'].values.reshape(-1, 1)  
y = df_yearly['ethnicity_score'].values  

#linear regression model and fit 
model = LinearRegression()
model.fit(X, y)

#predict the gender score values using the model
y_pred = model.predict(X)

# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(df_yearly['Movie Release Date'], df_yearly['ethnicity_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5, label='Mean ethnicity score')

#plot reg line
plt.plot(df_yearly['Movie Release Date'], y_pred, color='r', linestyle='--', linewidth=2, label='Linear Regression line')

#add regression equation 
equation_text = f'Ethnicity Score = {model.coef_[0]:.4f} * Year + {model.intercept_:.2f}'
plt.text( df_yearly['Movie Release Date'].min(), max(y) * 0.85, equation_text, fontsize=8, color='darkred')

# Add labels and title
plt.xlabel('Movie release year', fontsize=14)
plt.ylabel('Mean ethnicity score', fontsize=14)
plt.title('Mean ethnicity score by year with LR fit', fontsize=16)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#show explicitly the regression model's coefficients and intercept
print(f'Linear regression coefficient: {model.coef_[0]}')
print(f'Linear regression intercept: {model.intercept_}')



df_yearly_gender = Diversity_movie_metadata.groupby('Movie Release Date')['gender_score'].mean().reset_index()
df_yearly_ethnicity = Diversity_movie_metadata.groupby('Movie Release Date')['ethnicity_score'].mean().reset_index()
df_yearly_age = Diversity_movie_metadata.groupby('Movie Release Date')['age_score'].mean().reset_index()
df_yearly_foreign_actor = Diversity_movie_metadata.groupby('Movie Release Date')['Foreign Actor Proportion'].mean().reset_index()
df_yearly_diversitymean = Diversity_movie_metadata.groupby('Movie Release Date')['diversity_score'].mean().reset_index()
df_yearly_height = Diversity_movie_metadata.groupby('Movie Release Date')['height_score'].median().reset_index()
#make a list of dfs
dfs = [
    ('Gender Score', df_yearly_gender), 
    ('Ethnicity Score', df_yearly_ethnicity), 
    ('Age Score', df_yearly_age), 
    ('Foreign Actor Proportion', df_yearly_foreign_actor), 
    ('Diversity Score', df_yearly_diversitymean), 
    ('Height Score', df_yearly_height)
]

#perform linear regression over every df in the list and store the results in a list
results = []
for label, df in dfs:
    X = df['Movie Release Date'].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    results.append((label, model.coef_[0], model.intercept_))

#make a barplot of the coefficients with on the x axis the dfs and on the y the coefficients
labels = [result[0] for result in results]
coefficients = [result[1] for result in results]

plt.figure(figsize=(12, 6))
plt.bar(labels, coefficients, color='skyblue', alpha=0.7)
plt.xlabel('Diversity type', fontsize=14)
plt.ylabel('Regression coefficient (slope)', fontsize=14)
plt.title('Linear regression coefficients for different diversity metrics over time', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#importation
from sklearn.preprocessing import PolynomialFeatures

X = df_yearly_ethnicity['Movie Release Date'].values.reshape(-1, 1)  
y = df_yearly_ethnicity['ethnicity_score'].values  


degree = 3
# polynomial feature creation with defined degree
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)


#create linear regression model and fit it into the poly features
model = LinearRegression()
model.fit(X_poly, y)


#prediction
y_pred_poly = model.predict(X_poly)


#plot original data
plt.figure(figsize=(12, 6))
plt.plot(df_yearly_ethnicity['Movie Release Date'], df_yearly_ethnicity['ethnicity_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5, label='Mean ethnicity score')


#plot polynomial reg curve
plt.plot(df_yearly_ethnicity['Movie Release Date'], y_pred_poly, color='r', linestyle='--', linewidth=2, label= f"Polynomial regression curve degree, {degree}")


#add regression equation
coeffs = model.coef_
intercept = model.intercept_
equation_text = f"y = {intercept:.3f} + " + " + ".join([f"{coeffs[i]:.3f}x^{i}" for i in range(1, len(coeffs))])
plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')


plt.xlabel('Movie release year', fontsize=14)
plt.ylabel('Mean ethnicity score', fontsize=14)
plt.title('Mean ethnicity score by year with polynomial regression', fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#show explicitly the model's coefficients and intercept
print(f'Polynomial regression coefficients: {coeffs}')
print(f'Polynomial regression intercept: {intercept}')



#analysis of critical and inflection points
degree = 3  
X = df_yearly_ethnicity['Movie Release Date'].values
y = df_yearly_ethnicity['ethnicity_score'].values
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X.reshape(-1, 1))

#fit
model = LinearRegression()
model.fit(X_poly, y)

#get coeffs
coefficients = model.coef_
intercept = model.intercept_
poly_coeff = np.append(intercept, coefficients[1:]) 
p = np.poly1d(poly_coeff[::-1]) 

#derivatives for critical and inflection points
p_prime = p.deriv()  #first derivative
p_double_prime = p_prime.deriv()  #second derivative

# Find critical and inflection points
critical_points = p_prime.r
inflection_points = p_double_prime.r

#plot the polynomial curve and highlight critical and inflection points
x_vals = np.linspace(X.min(), X.max(), 500)
y_vals = p(x_vals)

plt.figure(figsize=(12, 6))
plt.plot(X, y, 'bo', label='Data')
plt.plot(x_vals, y_vals, 'r-', label=f'Polynomial regression (degree {degree})')
plt.scatter(critical_points, p(critical_points), color='g', s=100, zorder=5, label='Critical points')
plt.scatter(inflection_points, p(inflection_points), color='purple', s=100, zorder=5, label='Inflection points')
plt.xlabel('Movie release year')
plt.ylabel('Mean ethnicity score')
plt.title('Critical and Inflection Points of Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Print critical and inflection points
print("Critical Points (x values):", critical_points)
print("Inflection Points (x values):", inflection_points)


X = df_yearly_diversitymean['Movie Release Date'].values.reshape(-1, 1)  
y = df_yearly_diversitymean['diversity_score'].values  


degree = 6
# polynomial feature creation with defined degree
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)


#create linear regression model and fit it into the poly features
model = LinearRegression()
model.fit(X_poly, y)
#prediction
y_pred_poly = model.predict(X_poly)


#plot original data
plt.figure(figsize=(12, 6))
plt.plot(df_yearly_diversitymean['Movie Release Date'], df_yearly_diversitymean['diversity_score'], marker='o', color='b', linestyle='-', linewidth=2, markersize=5, label='Mean diversity score')


#plot polynomial reg curve
plt.plot(df_yearly_diversitymean['Movie Release Date'], y_pred_poly, color='r', linestyle='--', linewidth=2, label= f"Polynomial regression curve degree, {degree}")


#add regression equation
coeffs = model.coef_
intercept = model.intercept_
equation_text = f"y = {intercept:.3f} + " + " + ".join([f"{coeffs[i]:.3f}x^{i}" for i in range(1, len(coeffs))])
plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')


plt.xlabel('Movie release year', fontsize=14)
plt.ylabel('Mean diversity score', fontsize=14)
plt.title('Mean diversity score by year with polynomial regression', fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#show explicitly the model's coefficients and intercept
print(f'Polynomial regression coefficients: {coeffs}')
print(f'Polynomial regression intercept: {intercept}')



#pairwise relationships between diversity scores
sns.pairplot(df_merged[['age_score', 'height_score', 'ethnicity_score', 'gender_score', 'Foreign Actor Proportion', 'diversity_score']])
plt.suptitle('Pairwise relationship between diversity scores', y=1.02)
plt.tight_layout()
plt.show()
