from scipy.stats import chi2_contingency
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import os


current_path = os.getcwd()

# Loading the dataset
df_metadata_OI_exploded = pd.read_csv('data/metadata_OI_exploded.csv')
df_metadata_OI_compact = pd.read_csv('data/metadata_OI_compact.csv')


# Function to safely evaluate a string and convert it to a list
def safe_eval(val):
    if isinstance(val, list):
        return val
    else:
        val = val.strip()
        if val.startswith('["') or val.startswith("['"):
            val = val.replace("['", '["').replace("']", '"]')  # Normalize to double quotes
        return ast.literal_eval(val)

# Function to convert string representation of lists to actual lists for multiple columns
def convert_string_to_list(df, column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].apply(safe_eval)
    return df

# Function to count occurrences of each string in the 'Actor Country of Origin' column
def count_occurrences_in_column(df, column_name):
    all_values = []
    
    # Flatten the lists and count only string occurrences
    for sublist in df[column_name]:
        if isinstance(sublist, list):
            for item in sublist:
                if isinstance(item, str) and item.strip():  # Ensure it's a non-empty string
                    # Split string entries that have multiple countries, e.g., "India', 'Pakistan"
                    countries = item.split(",")  # Split by commas
                    for country in countries:
                        country = country.strip().strip("'").strip('"')  # Clean the string
                        if country:
                            all_values.append(country)
    return Counter(all_values)


columns_to_convert = ['Movie Language', 'Movie Country', 'Actor Ethnicity', 'Actor Country of Origin']
df_metadata_OI_compact = convert_string_to_list(df_metadata_OI_compact, columns_to_convert)


df_gender_value_count = df_metadata_OI_compact['Actor Gender'].value_counts()

#calculate the total counts
total_count = sum(df_gender_value_count)

print(df_gender_value_count)

# Create a bar plot
df_gender_value_count.plot(kind='bar')

# Add value labels on top of each bar
for i, value in enumerate(df_gender_value_count):
    percentage = (value / total_count) * 100
    plt.text(i, value + max(df_gender_value_count) * 0.01, f'{percentage:.1f}%', ha='center')
    
# Customize the plot (optional)
plt.title('Gender Counts')
plt.xlabel('Gender')
plt.ylabel('Frequency')

# Show the plot
plt.show()


#create a boxplot to show the distribution of Actor Height by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Actor Gender', y='Actor Height', data=df_metadata_OI_compact)

for gender in df_metadata_OI_compact['Actor Gender'].unique():
    median = df_metadata_OI_compact[df_metadata_OI_compact['Actor Gender'] == gender]['Actor Height'].median()
    std_dev = df_metadata_OI_compact[df_metadata_OI_compact['Actor Gender'] == gender]['Actor Height'].std()
    
    #add the median and std on the plot
    plt.text(gender, median + 0.05, f'Median: {median:.2f}', horizontalalignment='center')
    plt.text(gender, median - 0.05, f'Std Dev: {std_dev:.2f}', horizontalalignment='center')

plt.title('Actor Height Distribution by Gender')
plt.xlabel('Actor Gender')
plt.ylabel('Actor Height')

plt.show()


# Get the height counts and sort by the index (height values)
df_height_sorted = df_metadata_OI_compact['Actor Height'].value_counts().sort_index()
total_count_height = sum(df_height_sorted )

# Create a bar plot with sorted heights
df_height_sorted.plot(kind='bar')

# Add value labels on top of each bar
for i, value in enumerate(df_height_sorted):
    percentage = (value / total_count_height) * 100
    plt.text(i, value + max(df_height_sorted) * 0.01, f'{percentage:.1f}%', ha='center')

# Customize the plot (optional)
plt.title('Height Distribution')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# Bin ages in 5-year intervals
age_bins = pd.cut(df_metadata_OI_compact['Actor Age'], bins=range(0, int(df_metadata_OI_compact['Actor Age'].max()) + 5, 5))
total_count_age = sum(age_bins.value_counts())

# Count occurrences in each age bin
df_age_binned = age_bins.value_counts().sort_index()

# Plot the binned ages
df_age_binned.plot(kind='bar', width=0.8)

for i, value in enumerate(df_age_binned):
    percentage = (value / total_count_age) * 100
    plt.text(i, value + max(df_age_binned) * 0.01, f'{percentage:.1f}%', ha='center', fontsize=7)

# Customize the plot (optional)
plt.title('Age Distribution in 5-Year Intervals')
plt.xlabel('Age Range')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# Show the plot
plt.show()


# Count the occurrences of ethnicities
ethnicity_value_count = count_occurrences_in_column(df_metadata_OI_compact, 'Actor Ethnicity')

# Convert the Counter object into a DataFrame
df_ethnicity_value_count = pd.DataFrame(ethnicity_value_count.items(), columns=['Ethnicity', 'Count'])

# Set 'Ethnicity' as the index
df_ethnicity_value_count.set_index('Ethnicity', inplace=True)

print(df_ethnicity_value_count)



# Decide a threshold for the minimum count to display
threshold = 30
other_ethnicities = df_ethnicity_value_count[df_ethnicity_value_count['Count'] >= threshold].index

# Create a copy of the df_metadata_OI to not modify the original one
df_metadata_OI_copy = df_metadata_OI_compact.copy()

# Apply the logic to the 'Actor Ethnicity' column
df_metadata_OI_copy['Actor Ethnicity'] = df_metadata_OI_copy['Actor Ethnicity'].apply(
    lambda x: [ethnicity if ethnicity in other_ethnicities else 'Other' for ethnicity in x] if isinstance(x, list) else ('Other' if x not in other_ethnicities else x)
)

# Flatten the lists in the 'Actor Ethnicity' column for plotting
df_metadata_OI_copy_exploded = df_metadata_OI_copy.explode('Actor Ethnicity')

# Plot the count of ethnicities after regrouping
plt.figure(figsize=(12, 10))
sns.countplot(data=df_metadata_OI_copy_exploded, y='Actor Ethnicity', order=df_metadata_OI_copy_exploded['Actor Ethnicity'].value_counts().index, color='orange')

# Add percentages at the end of each bar
total = len(df_metadata_OI_copy_exploded)
ax = plt.gca()
for p in ax.patches:
    percentage = f'{100 * p.get_width() / total:.1f}%'
    plt.text(p.get_width() + 5, p.get_y() + p.get_height() / 2, percentage, ha='left', va='center')

plt.title('Count (percentage) of actors by ethnicity with Other regrouping ethnicities of less than 70 counts')
plt.xlabel('Count')
plt.ylabel('Ethnicity')
plt.show()



# Count the occurrences of countries in 'Actor Country of Origin'
origin_value_count = count_occurrences_in_column(df_metadata_OI_compact, 'Actor Country of Origin')

# Convert the Counter object into a DataFrame
df_origin_value_count = pd.DataFrame(origin_value_count.items(), columns=['Country', 'Count'])

# Set 'Country' as the index
df_origin_value_count.set_index('Country', inplace=True)

print(df_origin_value_count)



# Decide a threshold for the minimum count to display
threshold = 30
other_countries = df_origin_value_count[df_origin_value_count['Count'] >= threshold].index

# Create a copy of the df_metadata_OI to not modify the original one
df_metadata_OI_copy = df_metadata_OI_compact.copy()

# Apply the logic to the 'Actor Country of Origin' column
df_metadata_OI_copy['Actor Country of Origin'] = df_metadata_OI_copy['Actor Country of Origin'].apply(
    lambda x: [country if country in other_countries else 'Other_countries' for country in x] 
    if isinstance(x, list) 
    else ('Other_countries' if x not in other_countries else x)
)

# Flatten the lists in the 'Actor Country of Origin' column for plotting
df_metadata_OI_copy_exploded = df_metadata_OI_copy.explode('Actor Country of Origin')

# Plot the count of countries after regrouping
plt.figure(figsize=(12, 10))
sns.countplot(data=df_metadata_OI_copy_exploded, y='Actor Country of Origin', order=df_metadata_OI_copy_exploded['Actor Country of Origin'].value_counts().index, color='orange')

# Add percentages at the end of each bar
total = len(df_metadata_OI_copy_exploded)
ax = plt.gca()
for p in ax.patches:
    percentage = f'{100 * p.get_width() / total:.1f}%'
    plt.text(p.get_width() + 5, p.get_y() + p.get_height() / 2, percentage, ha='left', va='center')

plt.title(f'Count (percentage) of actors by countries with Other regrouping countries of less than {threshold} counts')
plt.xlabel('Count')
plt.ylabel('Country of Origin')
plt.show()



