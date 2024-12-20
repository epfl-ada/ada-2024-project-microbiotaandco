# This Script is used to analyse the diversity and the Box office revenue of the movies.
# We used different diversity scores
# - age_score
# - height_score
# - ethnicity_score
# - Foreign Actor Proportion
# - gender_score


from scipy.stats import chi2_contingency
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import os
import warnings
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


warnings.filterwarnings("ignore")



# Load the data
df_metadata_OI_exploded=pd.read_csv('data/metadata_OI_exploded.csv')

# Extract the unique values of the columns 'Movie Country', 'Movie Language', 'Movie Release Date', 'Movie Box Office Revenue'
movie_info = df_metadata_OI_exploded.groupby('Freebase Movie ID').agg({
    'Movie Country': lambda x: list(x.unique()),  
    'Movie Language': lambda x: list(x.unique()), 
    'Movie Release Date': 'first',                
    'Movie Box Office Revenue': 'first'     }).reset_index()

# Merge the dataframes
df_merged_1 = df_metadata_OI_exploded.drop(columns=['Movie Country', 'Movie Language', 'Movie Release Date', 'Movie Box Office Revenue']) \
                          .merge(movie_info, on='Freebase Movie ID', how='left')

# Select the columns of interest
df_merged_1 = df_merged_1[[
    'Freebase Movie ID', 'Movie Country', 'Movie Language', 'Movie Release Date', 'Movie Box Office Revenue',
    'Actor Age', 'Actor Gender', 'Actor Ethnicity', 'Actor Height', 'Actor Country of Origin']]

# Drop duplicates
df_merged_unique = df_merged_1.drop_duplicates(subset=[
    'Freebase Movie ID', 'Actor Height', 'Actor Ethnicity', 'Actor Age', 'Actor Gender', 'Actor Country of Origin'
], keep='first')

# count = df_merged_unique['Freebase Movie ID'].value_counts().get('/m/011yfd', 0)

# Find the movies which appear more than once
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




# Calculate diversity scores gender
df_result_gender = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_gender_diversity).reset_index()
df_result_gender.columns = ['Freebase Movie ID', 'gender_score']



#  Calculate the proportion of foreign actors in each movie
df_result_foreigners = calculate_foreign_actor_proportion(df_metadata_OI)



# Calculate diversity scores age
df_result_age = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_age_diversity).reset_index()
df_result_age.columns = ['Freebase Movie ID', 'age_score']


# Calculate diversity scores for height
df_result_height = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_height_diversity).reset_index()
df_result_height.columns = ['Freebase Movie ID', 'height_score']


# Calculate diversity scores for ethnicity
df_result_ethnicity = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_ethnicity_diversity).reset_index()
df_result_ethnicity.columns = ['Freebase Movie ID', 'ethnicity_score']


# merge the diversity scores
df_merged = df_result_age \
    .merge(df_result_height, on='Freebase Movie ID') \
    .merge(df_result_ethnicity, on='Freebase Movie ID') \
    .merge(df_result_gender, on='Freebase Movie ID')\
    .merge(df_result_foreigners, on='Freebase Movie ID')


# Calculate the diversity score
df_merged['diversity_score'] = df_merged[['age_score', 'height_score', 'ethnicity_score', 'gender_score','Foreign Actor Proportion']].mean(axis=1)
df_merged.head(10)


# Merge the diversity scores with the movie metadata
Diversity_movie_metadata=df_merged.merge(
    df_merged_unique[['Freebase Movie ID', 'Movie Release Date', 'Movie Box Office Revenue', 'Movie Language', 'Movie Country']],
    on='Freebase Movie ID',
    how='inner') 


from sklearn.linear_model import LassoCV






# Drop duplicates movies
Diversity_movie_metadata=Diversity_movie_metadata.drop_duplicates(subset=["Freebase Movie ID"])


# 
model_boxoffice_date=LassoCV(cv=5, random_state=0)

# show the first 5 rows of the data
(Diversity_movie_metadata.head())


# load movie metadata
df_full = pd.read_csv("./src/data/movie_metadata.csv")
df_full.dropna(inplace=True)

Diversity_movie_metadata_completed=Diversity_movie_metadata.merge(
    df_full[['Freebase Movie ID', 'Movie Runtime','Movie Genre']],
    on='Freebase Movie ID',
    how='inner') 


# Create dummy variables for the movie language and movie country
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
dummy_categories = pd.DataFrame(mlb.fit_transform(Diversity_movie_metadata_completed['Movie Language']), columns=mlb.classes_)
dummy_categories_2 = pd.DataFrame(mlb.fit_transform(Diversity_movie_metadata_completed['Movie Country']), columns=mlb.classes_)
dummy_categories.columns=["Language: "+col for col in dummy_categories.columns]
dummy_categories_2.columns=["Movie: "+col for col in dummy_categories_2.columns]

# Create dummy variables for the movie genre
def parse_if_string(value):
    if isinstance(value, str):
        try:
            # Try to parse the string as a dictionary
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value  
    return value  

Diversity_movie_metadata_completed['genre categories'] = Diversity_movie_metadata_completed['Movie Genre'].apply(parse_if_string)
categories_expanded = Diversity_movie_metadata_completed['genre categories'].apply(lambda x:list(x.values()))
dummy_categories_3=pd.DataFrame(mlb.fit_transform(categories_expanded), columns=mlb.classes_)
dummy_categories_3.columns=["Genre: "+col for col in dummy_categories_3.columns]

# Concatenate the dataframes
df_dummy = pd.concat([Diversity_movie_metadata_completed.drop(['Movie Language', 'Movie Genre','Movie Country'], axis=1), dummy_categories, dummy_categories_3,dummy_categories_2], axis=1)

df=df_dummy.copy()

limit_of_dummy=30
for col in list(dummy_categories.columns):
    try:
        if df[col].sum()<limit_of_dummy:
            df.drop(columns=col,inplace=True)
    except KeyError:
        _=1     
for col in list(dummy_categories_2.columns):
    try:
        if df[col].sum()<limit_of_dummy:
            df.drop(columns=col,inplace=True)
    except KeyError:
        s=1    
for col in list(dummy_categories_3.columns):
    try:
        if df[col].sum()<limit_of_dummy:
            df.drop(columns=col,inplace=True)
    except KeyError:
        s=1 



            #Drop Outliers for the value of the Box Office Revenue

column_of_interest = "Movie Box Office Revenue"

# Compute the Q1 (25th percentile) and Q3 (75th percentile) for the specific column
Q1 = df[column_of_interest].quantile(0.25)
Q3 = df[column_of_interest].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define the outlier condition for the specific column
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows where the value in the specific column is an outlier
df_cleaned = df[(df[column_of_interest] >= lower_bound) & (df[column_of_interest] <= upper_bound)]

df=df_cleaned.copy()


column_of_interest = "Movie Box Office Revenue"

# Compute the Q1 (25th percentile) and Q3 (75th percentile) for the specific column
Q1 = df[column_of_interest].quantile(0.25)
Q3 = df[column_of_interest].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define the outlier condition for the specific column
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows where the value in the specific column is an outlier
df_cleaned = df[(df[column_of_interest] >= lower_bound) & (df[column_of_interest] <= upper_bound)]

df=df_cleaned.copy()

# define your Lasso cross-validation model
custom_alphas = np.linspace(1e-4, 1e-2, 1001)
model_cv_lasso_1 = LassoCV(cv = 5, random_state = 42, max_iter = 10000, alphas=custom_alphas)

# define the target and the explanatory variables (and also the non-dummy columns)
non_dummy_columns=["age_score", "height_score", "ethnicity_score", "gender_score", "Foreign Actor Proportion",  "Movie Release Date", "Movie Runtime"]

explanatory_columns=non_dummy_columns+(list(dummy_categories.columns))
explanatory_columns=explanatory_columns+(list(dummy_categories_3.columns))
explanatory_columns=explanatory_columns+(list(dummy_categories_2.columns))
explanatory_columns = [col for col in explanatory_columns if col in df.columns]

target="Movie Box Office Revenue"

# define the target and the explanatory vector and matrix
X=df[explanatory_columns]
y=np.log(df[target]/(df[target].mean()))


#define model LASSO with CV and alphas range
custom_alphas = np.linspace(1e-4, 1e-2, 1001)
model_cv_lasso_1 = LassoCV(cv = 5, random_state = 42, max_iter = 10000, alphas=custom_alphas)


# standardize the non-dummy columns
X_dd=X.copy()
scaler_dd = StandardScaler()
X_dd[non_dummy_columns]=scaler_dd.fit_transform(X_dd[non_dummy_columns])


# fit the model
model_cv_lasso_1.fit(X_dd, y)
#model_cv_lasso_1.coef_
print("Lasso was used to select the variable of interest.")
alpha_1 = model_cv_lasso_1.alpha_
# define the non-zero features
non_zero_features = X.columns[model_cv_lasso_1.coef_ != 0]


# define the model OLS with the non-zero features and output the summary of the estimation
modelOLS_total=sm.OLS(y,sm.add_constant(X_dd[non_zero_features])).fit()

"""
print("OLS was used to estimate the effect of the selected variables.")
print("Here is the summary of the estimated model:")
print(modelOLS_total.summary())
"""


from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline




               # Plot the Lasso coefficients as a function of the regularization strength
print("The Lasso coefficients are plotted as a function of the regularization strength.")
alphas = np.logspace(-4, 0, 50)
feature_names_2 = ["gender_score",
    "age_score",
    "height_score",
    "ethnicity_score",
    
    "Foreign Actor Proportion"
]

# dictioannary to rename the features in the plot
feature_names_2_dict = {"gender_score": "gender diversity coefficient", "age_score": "age diversity coefficient", "height_score": "height diversity coefficient", "ethnicity_score": "ethnicity diversity coefficient", "Foreign Actor Proportion": "foreign actor proportion coefficient"} 

coefficients = []
list_of_non_zero_coefficients = []
for alpha in alphas:
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha))
    lasso.fit(X_dd, y)
    coefficients.append(lasso.named_steps['lasso'].coef_)
    # count the number of non-zero coefficients
    non_zero_coefficients = np.sum(lasso.named_steps['lasso'].coef_ != 0)
    list_of_non_zero_coefficients.append(non_zero_coefficients)
coefficients = np.array(coefficients)

plt.figure(figsize=(10, 6))
colors = ['thistle', 'palegoldenrod', 'lightsalmon', 'mediumseagreen', 'lightblue']
for i, feature in enumerate(feature_names_2):
    plt.plot(alphas, coefficients[:, i],  color=colors[i],label=feature_names_2_dict[feature])
    plt.fill_between(alphas, coefficients[:, i], 0, alpha=0.2,color=colors[i])



plt.xscale("log")
plt.xlabel("Regularization Strength (alpha)")
plt.ylabel("Coefficient Value")
plt.title("Lasso Coefficients as a Function of Regularization Strength")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

#draw vertical line at optimal alpha
plt.axvline(alpha_1, color='red', linestyle='--', linewidth=0.8,label=f"Optimal Alpha for CV LASSO")
# add legend to the bottom left of the plot
plt.legend(loc='lower left')


plt.twinx()
plt.plot(alphas, list_of_non_zero_coefficients, label='Number of non-zero Coefficients', color='black', linestyle='--')
plt.ylabel("Number of Non-zero Coefficients")
#add a vertical line at when the number of non-zero coefficients is 10
index_10 = next(i for i, x in enumerate(list_of_non_zero_coefficients) if x <= 12)
plt.axvline(alphas[index_10], color='black', linestyle='--', linewidth=0.8, label='alpha such that: Number of non-zero Coefficients = 9')

plt.legend(loc='upper right') 
plt.show()


print("But have we really solved colinearity issues?")


corr_matrix = X_dd[non_zero_features].corr()
mask = (corr_matrix.abs() >= 0.4) & (corr_matrix != 1.0)

# Identify rows and columns with at least one high correlation
filtered_indices = mask.any(axis=1)

# Filter the correlation matrix
filtered_corr_matrix = corr_matrix.loc[filtered_indices, filtered_indices]

# heatmap
'''
plt.figure(figsize=(10, 6))
sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', cbar=True)

plt.title('Correlation Heatmap of diversity scores', fontsize=16)
plt.tight_layout()
plt.show()
'''

# Erase some columns with high colinearity
print("LASSO was unable to get rid completely of high correlations.")
print("To solve this we propose the following: When there is a correlation > 0.4, we delete a variable.")
erase_column=[]
indices=np.where(mask)
pairs = [(i, j) for i, j in zip(indices[0], indices[1]) if i <= j]
for p in pairs:
    dummy1=corr_matrix.columns[p[0]]
    dummy2=corr_matrix.columns[p[1]]
    coef1=modelOLS_total.params[dummy1]
    coef2=modelOLS_total.params[dummy2]
    if coef1<coef2:
        if dummy1 not in erase_column:
            erase_column.append(dummy1)
    else:
        if dummy2 not in erase_column:
            erase_column.append(dummy2)
  
# This is the list of the non-zero features
updated_list_nocol_nonzero = [item for item in non_zero_features if item not in erase_column]

       
        # We can finally start the residual analysis/regression

# we define the features to compare (diversities scores and Box Office revenue) and the control variables
feature_to_compare=["age_score","height_score","ethnicity_score", "gender_score", "Foreign Actor Proportion","Movie Box Office Revenue"]
control_variable = [item for item in updated_list_nocol_nonzero if item not in feature_to_compare]

residuals = pd.DataFrame(columns=feature_to_compare)


# For each variable in (feature_to_compare), we regress it on the control variable and compute the residuals
for feat in feature_to_compare:
    if feat=="Movie Box Office Revenue":
        y_for_residuals=y.copy()
    else:
        y_for_residuals=X_dd[feat]
    modelOLS_residuals=sm.OLS(y_for_residuals,sm.add_constant(X_dd[control_variable])).fit()
    #print(modelOLS_residuals.rsquared)
    residuals[feat]=modelOLS_residuals.resid
print("Estimate the resiudals of the diversity scores and the Box Office Revenue on the control variables.")

feature_to_compare.remove("Movie Box Office Revenue")

# We regress the residuals of the Box Office Revenue on the residuals of the diversity scores
X_final=residuals[feature_to_compare]
y_final=residuals["Movie Box Office Revenue"]
modelOLS_residuals=sm.OLS(y_final,(X_final)).fit()
print("'''")
print("We now can plot the residuals of the Box Office Revenue against the residuals of the diversity scores.")
print("And obtain the summary of the regression:")
print((modelOLS_residuals.summary()))

import seaborn as sns
sns.kdeplot(y_final, fill=True, color='red', label='True residuals of the Box Office Revenue')
sns.kdeplot(modelOLS_residuals.predict(X_final), fill=True, color='blue', label='Estimated residuals of the Box Office Revenue')
plt.title('Distribution of estimated and true residuals of the Box Office Revenue')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.show()


# obtain the qq plot of the residuals
def plot_residuals_vs_fitted():
    residuals_qq = modelOLS_residuals.resid
    
    # Q-Q Plot
    sm.qqplot(residuals_qq, line='45', fit=True)
    plt.title('Q-Q Plot of Residuals')
    plt.show()


def plot_each_against(df, col_a, cols_b, xlabel=None, ylabel=None, colors=None):
    """
    Plots a single column A against each column in B from a DataFrame,
    creating one new scatter plot for each column in B.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col_a (str): The column name for the x-axis.
    - cols_b (list of str): List of column names to be plotted on the y-axis.
    - xlabel (str): Label for the x-axis (optional).
    - ylabel (str): Label for the y-axis (optional).
    - colors (list of str): List of colors for each scatter plot (optional).

    Returns:
    - None
    """
    for i, col_b in enumerate(cols_b):
        plt.figure(figsize=(8, 6))
        
        # Define color for each scatter plot
        color = colors[i] if colors and i < len(colors) else None  # Default color
        
        # Create the scatter plot for col_b
        plt.scatter( df[col_b],df[col_a], color=color, label=f'{col_a} vs {col_b}')
        
        # Add labels, legend, and title
        plt.xlabel(xlabel if xlabel else col_b)
        plt.ylabel(ylabel if ylabel else col_a)
        plt.title(f'{col_a} vs {col_b}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Display the plot
        plt.show()


# this plot is used to plot the residuals against the explanantory variables
def residuals_against_columns():

    ndf=(df.copy())
    residuals_qq = modelOLS_residuals.resid

# Calculate standard deviation of residuals
    residuals_std = np.std(residuals_qq, ddof=1)  # ddof=1 for sample standard deviation

# Calculate standardized residuals
    standardized_residuals = residuals_qq / residuals_std
    ndf["final residuals"]=standardized_residuals

    list_of_columns_for_plots=X.columns

    plot_each_against(ndf, "final residuals",list_of_columns_for_plots)





