import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr


Diversity_movie_metadata=pd.read_csv('data/Diversity_movie_metadata.csv')



median_diversity_score = Diversity_movie_metadata['diversity_score'].median()

Diversity_movie_metadata['diverse'] = (Diversity_movie_metadata['diversity_score'] > median_diversity_score).astype(int)


#differentiate above and under median as diverse and non diverse
treated = Diversity_movie_metadata[Diversity_movie_metadata['diverse']==1]
control = Diversity_movie_metadata[Diversity_movie_metadata['diverse']==0]
#get statistics of both datasets
treated['Movie Box Office Revenue'].describe()


control['Movie Box Office Revenue'].describe()
#get observable ratio
median_diverse = treated['Movie Box Office Revenue'].median()
median_non_diverse = control['Movie Box Office Revenue'].median()
observed_or=median_diverse/median_non_diverse
print(observed_or)

#correlation between diversity score and revenue
correlation, p_value = pearsonr(Diversity_movie_metadata['Movie Box Office Revenue'], Diversity_movie_metadata['diversity_score'])

print("Pearson correlation:", correlation)
print("P-value:", p_value)





# Combine treated and control revenue data to compute shared bin edges
combined_data = np.concatenate([treated['Movie Box Office Revenue'], control['Movie Box Office Revenue']])
bin_edges = np.histogram_bin_edges(combined_data, bins=30)

# Find the maximum revenue values
max_treated = treated['Movie Box Office Revenue'].max()
max_control = control['Movie Box Office Revenue'].max()

# Plot the histograms with the shared bins
plt.figure(figsize=(10, 6))
ax = sns.histplot(treated['Movie Box Office Revenue'], bins=bin_edges, kde=True, stat='density', color='blue', label='Treated')
sns.histplot(control['Movie Box Office Revenue'], bins=bin_edges, kde=True, stat='density', color='orange', label='Control', ax=ax)

# Add vertical lines for max values
plt.axvline(max_treated, color='blue', linestyle='--', linewidth=1.5, label=f'Max Treated: {max_treated:.2f}')
plt.axvline(max_control, color='orange', linestyle='--', linewidth=1.5, label=f'Max Control: {max_control:.2f}')

# Add titles and labels
ax.set(title='Movie Revenue Distribution', xlabel='Movie Revenue', ylabel='Income Density')
plt.legend()

# Save the plot as an image
plt.savefig('Movie_Revenue_Distribution_with_Max.png', dpi=300)

# Display the plot
plt.show()

Diversity_movie_metadata.boxplot(by='diverse', column='Movie Release Date', figsize = [5, 5], grid=True)
# Save the plot as an image
plt.savefig('Movie_Release.png', dpi=300)

plt.show()



# Check if 'treated' DataFrame exists
if 'treated' not in locals():
    raise ValueError("Ensure the 'treated' DataFrame is defined before running this code.")

# Add 'Language Count' column
treated['Language Count'] = treated['Movie Language'].apply(
    lambda x: len(eval(x)) if isinstance(x, str) else len(x)
)

# Count how many movies have 'English' in the list of languages
english_count = treated['Movie Language'].apply(
    lambda x: 'English' in eval(x) if isinstance(x, str) else 'English' in x
).sum()

print(f"Number of movies that include English: {english_count}")

# Calculate proportions for the bar plot
language_counts = treated['Language Count'].value_counts().sort_index()
proportions = language_counts / language_counts.sum()

# Plot proportions
plt.figure(figsize=(10, 6))
sns.barplot(x=proportions.index, y=proportions.values, color='skyblue')
plt.title('Proportion of Movies by Number of Languages')
plt.xlabel('Number of Languages')
plt.ylabel('Proportion of Movies')
plt.xticks(rotation=0)

plt.savefig('Movie_diverse_Nb_of_language.png', dpi=300)

plt.show()


# Check if 'treated' DataFrame exists
if 'control' not in locals():
    raise ValueError("Ensure the 'control' DataFrame is defined before running this code.")

# Add 'Language Count' column
control['Language Count'] = control['Movie Language'].apply(
    lambda x: len(eval(x)) if isinstance(x, str) else len(x)
)

# Count how many movies have 'English' in the list of languages
english_count = control ['Movie Language'].apply(
    lambda x: 'English' in eval(x) if isinstance(x, str) else 'English' in x
).sum()

print(f"Number of movies that include English: {english_count}")

# Calculate proportions for the bar plot
language_counts = control['Language Count'].value_counts().sort_index()
proportions = language_counts / language_counts.sum()

# Plot proportions
plt.figure(figsize=(10, 6))
sns.barplot(x=proportions.index, y=proportions.values, color='skyblue')
plt.title('Proportion of Movies by Number of Languages')
plt.xlabel('Number of Languages')
plt.ylabel('Proportion of Movies')
plt.xticks(rotation=0)
plt.savefig('Movie_no_Nb_of_language.png', dpi=300)
plt.show()


treated['Country Count'] = treated['Movie Country'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))


country_counts = treated['Country Count'].value_counts().sort_index()


proportions = country_counts / country_counts.sum()


plt.figure(figsize=(10, 6))
sns.barplot(x=proportions.index, y=proportions.values, color='red')
plt.title('Proportion of Movies by Number of countries in diverse movies')
plt.xlabel('Number of countries')
plt.ylabel('Proportion of Movies')
plt.xticks(rotation=0)
plt.savefig('Movie_diverse_Nb_of_Countries.png', dpi=300)
plt.show()


control['Country Count'] = control['Movie Country'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))


country_counts = control['Country Count'].value_counts().sort_index()


proportions = country_counts / country_counts.sum()


plt.figure(figsize=(10, 6))
sns.barplot(x=proportions.index, y=proportions.values, color='red')
plt.title('Proportion of Movies by Number of countries in non-diverse movies')
plt.xlabel('Number of countries')
plt.ylabel('Proportion of Movies')
plt.xticks(rotation=0)
plt.savefig('Movie_no_Nb_of_countries.png', dpi=300)
plt.show()



diversity_movie_metadata_copy = Diversity_movie_metadata.copy()

# Ensure Diversity_movie_metadata_copy exists
if 'diversity_movie_metadata_copy' not in locals():
    raise ValueError("Ensure the 'Diversity_movie_metadata_copy' DataFrame is defined before running this code.")

# Create the 'has_english_language' column
diversity_movie_metadata_copy['has_english_language'] = diversity_movie_metadata_copy['Movie Language'].apply(
    lambda x: 'English' in eval(x) if isinstance(x, str) else 'English' in x
)

diversity_movie_metadata_copy['language_count'] = diversity_movie_metadata_copy['Movie Language'].apply(
    lambda x: len(eval(x)) if isinstance(x, str) else len(x)
)

# Add 'country_count' column
diversity_movie_metadata_copy['country_count'] = diversity_movie_metadata_copy['Movie Country'].apply(
    lambda x: len(eval(x)) if isinstance(x, str) else len(x)
)
# Check the resulting DataFrame
diversity_movie_metadata_copy.sample(50)




#propensity score


# Now you can work with 'diversity_movie_metadata_copy' without affecting the original DataFrame
diversity_movie_metadata_copy.rename(columns={'Movie Release Date': 'Movie_Release_Date'}, inplace=True)


# let's standardize the continuous features
diversity_movie_metadata_copy['Movie_Release_Date'] = (diversity_movie_metadata_copy['Movie_Release_Date'] - diversity_movie_metadata_copy['Movie_Release_Date'].mean())/diversity_movie_metadata_copy['Movie_Release_Date'].std()
diversity_movie_metadata_copy['language_count'] = (diversity_movie_metadata_copy['language_count'] - diversity_movie_metadata_copy['language_count'].mean())/diversity_movie_metadata_copy['language_count'].std()
diversity_movie_metadata_copy['country_count'] = (diversity_movie_metadata_copy['country_count'] - diversity_movie_metadata_copy['country_count'].mean())/diversity_movie_metadata_copy['country_count'].std()


mod = smf.logit(formula='diverse ~ Movie_Release_Date + language_count + country_count +C(has_english_language)', data=diversity_movie_metadata_copy)

res = mod.fit()

# Extract the estimated propensity scores
diversity_movie_metadata_copy['Propensity_score'] = res.predict()

print(res.summary())


def get_similarity(propensity_score1, propensity_score2):
    '''Calculate similarity for instances with given propensity scores'''
    return 1-np.abs(propensity_score1-propensity_score2)





treated = diversity_movie_metadata_copy[diversity_movie_metadata_copy['diverse'] == 1]
control = diversity_movie_metadata_copy[diversity_movie_metadata_copy['diverse'] == 0]


assert 'Propensity_score' in control.columns, "Propensity_score column is missing in the control DataFrame"
assert 'Propensity_score' in treated.columns, "Propensity_score column is missing in the treated DataFrame"

# Fit NearestNeighbors model on the control group
nn = NearestNeighbors(n_neighbors=5)
nn.fit(control[['Propensity_score']])

# Find nearest neighbors for treatment group
distances, indices = nn.kneighbors(treated[['Propensity_score']])

# Fit NearestNeighbors model on the control group
nn = NearestNeighbors(n_neighbors=5)  # Adjust `n_neighbors` as required
nn.fit(control[['Propensity_score']])

# Find nearest neighbors for treatment group
distances, indices = nn.kneighbors(treated[['Propensity_score']])

# Create an empty undirected graph
G = nx.Graph()

# Add edges for each treatment sample and its nearest neighbors
for treatment_id, (dist_list, idx_list) in enumerate(zip(distances, indices)):
    for dist, control_idx in zip(dist_list, idx_list):
        similarity = 1 / (1 + dist)  # Example similarity function
        # Node IDs must not overlap, adjust control indices to unique IDs
        control_node = control_idx + len(treated)  
        G.add_weighted_edges_from([(treatment_id, control_node, similarity)])

# Generate the maximum weight matching
matching = nx.max_weight_matching(G, maxcardinality=True)



# Print matching with propensity scores
print("Matching with Propensity Scores:")
for treatment_node, control_node in matching:
    if treatment_node < len(treated):  # Identify treatment and control nodes
        treatment_id = treatment_node
        control_id = control_node - len(treated)
    else:
        treatment_id = control_node
        control_id = treatment_node - len(treated)

    # Get propensity scores
    treatment_score = treated.iloc[treatment_id]['Propensity_score']
    control_score = control.iloc[control_id]['Propensity_score']

   # print(f"Treatment ID {treatment_id} (Propensity Score: {treatment_score:.4f}) "
      #    f"is matched to Control ID {control_id} (Propensity Score: {control_score:.4f})")
    
matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]

balanced_diversity_dataset = diversity_movie_metadata_copy.iloc[matched]

treated = balanced_diversity_dataset.loc[balanced_diversity_dataset['diverse'] == 1] 
control = balanced_diversity_dataset.loc[balanced_diversity_dataset['diverse'] == 0] 

treated['Movie Box Office Revenue'].describe()

control['Movie Box Office Revenue'].describe()




# Combine treated and control revenue data to compute shared bin edges
combined_data = np.concatenate([treated['Movie Box Office Revenue'], control['Movie Box Office Revenue']])
bin_edges = np.histogram_bin_edges(combined_data, bins=30)

# Find the maximum revenue values
max_treated = treated['Movie Box Office Revenue'].max()
max_control = control['Movie Box Office Revenue'].max()

# Plot the histograms with the shared bins
plt.figure(figsize=(10, 6))
ax = sns.histplot(treated['Movie Box Office Revenue'], bins=bin_edges, kde=True, stat='density', color='blue', label='Treated')
sns.histplot(control['Movie Box Office Revenue'], bins=bin_edges, kde=True, stat='density', color='orange', label='Control', ax=ax)

# Add vertical lines for max values
plt.axvline(max_treated, color='blue', linestyle='--', linewidth=1.5, label=f'Max Treated: {max_treated:.2f}')
plt.axvline(max_control, color='orange', linestyle='--', linewidth=1.5, label=f'Max Control: {max_control:.2f}')

# Add titles and labels
ax.set(title='Movie Revenue Distribution', xlabel='Movie Revenue', ylabel='Income Density')
plt.legend()

# Save the plot as an image
plt.savefig('Movie_New_Revenue_Distribution_with_Max.png', dpi=300)

# Display the plot
plt.show()


from scipy.stats import pearsonr

correlation, p_value = pearsonr(balanced_diversity_dataset['Movie Box Office Revenue'], balanced_diversity_dataset['diversity_score'])

print("Pearson correlation:", correlation)
print("P-value:", p_value)


def sensitivity_analysis(observed_or, gamma_values):
    """
    Perform a bounded odds ratio sensitivity analysis.
    
    Parameters:
    observed_or (float): Observed odds ratio from the study.
    gamma_values (list or array): Range of sensitivity parameter Γ to evaluate.

    Returns:
    pd.DataFrame: Sensitivity bounds for each Γ.
    """
  

    results = []
    for gamma in gamma_values:
        lower_bound = observed_or / gamma
        upper_bound = observed_or * gamma
        results.append({"Gamma": gamma, "Lower Bound": lower_bound, "Upper Bound": upper_bound})
    
    return pd.DataFrame(results)

# Example inputs
#observed_or = 2.5  # Observed odds ratio from your study
gamma_values = [1, 1.5, 2,2.5, 3, 5, 10]  # Range of Γ values to test

# Perform sensitivity analysis
sensitivity_results = sensitivity_analysis(observed_or, gamma_values)

# Display results
print(sensitivity_results)

