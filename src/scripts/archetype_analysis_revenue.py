import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

# import data
characters = pd.read_csv("src/data/characters_preprocessed_for_archetype_revenue.csv")
archetypes = pd.read_csv("src/data/Extracted_Data.csv")

archetypes = archetypes.rename(columns = {'Character_name' : 'Character Name', 'Freebase_movie_ID' : 'Freebase Movie ID', 'Actor_name' : 'Actor Name'})
df = pd.merge(characters, archetypes, how = 'inner', on = ["Freebase Movie ID", "Actor Name"])

#drop NAs in occupations labels and actor names
df = df.dropna(subset=["Actor Name", "Occupation Labels"])

#one hot encode the occupations
#Split the comma-separated strings into lists
df['Occupation Labels'] = df['Occupation Labels'].str.split(', ')

#Explode the lists so each occupation is in a separate row
df_exploded = df.explode('Occupation Labels').reset_index()  # Reset index to retain original row identifiers

#One-hot encode the 'Occupation Labels' column
occupation_dummies = pd.get_dummies(df_exploded['Occupation Labels'])

#Group by the original 'index' column and aggregate
df_one_hot_encoded = occupation_dummies.groupby(df_exploded['index']).max()

#Combine with the original DataFrame (excluding the exploded column to avoid duplication)
df_final = df.drop(columns='Occupation Labels').join(df_one_hot_encoded)

dummy_columns = df_one_hot_encoded.columns[df_one_hot_encoded.sum() > 50]
df_filtered = df.drop(columns='Occupation Labels').join(df_one_hot_encoded[dummy_columns])
df_filtered = df_filtered.dropna(subset=['Movie Box Office Revenue'])
df_filtered['Movie Release Date'] = df_filtered['Movie Release Date'].astype(str).str.replace('.0', '', regex=False)
df_filtered['Movie Release Date'] = pd.to_datetime(df_filtered['Movie Release Date'], format='%Y')

# Ensure 'Movie Release Date' is in datetime format
df_filtered['Movie Release Date'] = pd.to_datetime(df_filtered['Movie Release Date'])

# Extract the year for adjustment purposes (if necessary)
df_filtered['Release Year'] = df_filtered['Movie Release Date'].dt.year


# Linear regression of archetypes on residuals of release date predictin revenue

#get residuals:
X = df_filtered['Release Year']
y = df_filtered['Movie Box Office Revenue']
X = X.astype(float)
X_with_intercept = sm.add_constant(X)
model = sm.OLS(y, X_with_intercept)
results = model.fit()
predicted_revenue = results.predict(X_with_intercept)
residuals = y - predicted_revenue


# Select the one-hot encoded archetypes (columns 22 and onwards) and the adjusted revenue
X = df_filtered.iloc[:, 22:-2]  # Archetypes columns
y = residuals

# Fit the regression model
X = X.astype(int)  # Ensure X is numeric
X_with_intercept = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X_with_intercept)  # OLS regression
results = model.fit()  # Fit the model

# Print the summary of regression results
print(f'Linear regression of archetypes on residuals of release date prediction of revenue:')
print(results.summary())

# Extract p-values from the regression results
p_values = results.pvalues
p_values_archetype_movie_revenue = p_values.drop('const')

# Apply Bonferroni correction
m = len(p_values)  # Number of features (including the intercept)
bonferroni_p_values = p_values * m  # Bonferroni adjustment

# Correct p-values greater than 1 to 1 (since p-values cannot exceed 1)
#bonferroni_p_values = bonferroni_p_values.clip(upper=1.0)
sorted_bonferroni_p_values = bonferroni_p_values[bonferroni_p_values < 0.05].sort_values(ascending=True)
sorted_bonferroni_p_values_all = bonferroni_p_values.sort_values(ascending=True)

# Print the corrected p-values
print("Bonferroni Adjusted p-values:")
print(sorted_bonferroni_p_values.apply(lambda x: f'{x:.3f}'))
print("-"*50)


#graphs of chi2 pvalues v revenue p values
# gender

filtered_p_values = sorted_bonferroni_p_values_all.drop('const')
filtered_archetypes = filtered_p_values.index.tolist()
p_values = []
for archetype in filtered_archetypes:
    contingency_table = pd.crosstab(df_filtered['Actor Gender'], df_filtered[archetype])
    
    # Perform the Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    p_values.append(p)

plt.figure(figsize = (10,8))
plt.scatter(p_values_archetype_movie_revenue, p_values, label='Data points', color = 'tan')

X = sm.add_constant(p_values_archetype_movie_revenue)  # Adds the intercept (constant)

# Fit the linear regression for graph
model = sm.OLS(p_values, X) 
results = model.fit()

intercept = results.params['const']
slope = results.params[0]

# Get the p-value for the slope
slope_p_value = results.pvalues[0]

line = intercept + slope * p_values_archetype_movie_revenue 


# Plot the regression line
plt.plot(p_values_archetype_movie_revenue, line, color='red', label=f'Regression line: y = {slope:.3f}x + {intercept:.3f}\n P-value of the slope: {slope_p_value:.3f}')

# Add labels and legend
plt.title("How significantly disparities in gender diversity in archetypes impact the revenue")
plt.xlabel('P-values showing how significantly an archetype impacts the film revenue')
plt.ylabel('P-values showing how significantly the archetype has gender disparities')

# Show the plot
plt.legend()
plt.show()


# ethnicities

filtered_p_values = sorted_bonferroni_p_values_all.drop('const')
filtered_archetypes = filtered_p_values.index.tolist()
p_values = []
for archetype in filtered_archetypes:
    contingency_table = pd.crosstab(df_filtered['Actor Ethnicity'], df_filtered[archetype])
    
    # Perform the Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    p_values.append(p)

plt.figure(figsize = (10,8))
# Create the scatter plot
plt.scatter(p_values_archetype_movie_revenue, p_values, label='Data points', color = 'tan')

#-----------
X = sm.add_constant(p_values_archetype_movie_revenue)

# Fit the linear regression for graph
model = sm.OLS(p_values, X)
results = model.fit()

intercept = results.params['const']
slope = results.params[0]

# Get the p-value for the slope
slope_p_value = results.pvalues[0]

line = intercept + slope * p_values_archetype_movie_revenue 

# Plot the regression line
plt.plot(p_values_archetype_movie_revenue, line, color='red', label=f'Regression line: y = {slope:.3f}x + {intercept:.3f}\n P-value of the slope: {slope_p_value:.3f}')

# Add labels and legend
plt.title("How significantly disparities in ethnicity diversity in archetypes impact the revenue")
plt.xlabel('P-values showing how significantly an archetype impacts the film revenue')
plt.ylabel('P-values showing how significantly the archetype has ethnicity disparities')

# Show the plot
plt.legend()
plt.show()



