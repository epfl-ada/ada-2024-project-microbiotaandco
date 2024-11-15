from scipy.stats import chi2_contingency
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import os

def safe_eval(val):
    """
    Safely evaluates a string that represents a list and converts it into an actual Python list.
    
    This function is useful for handling string representations of lists in a controlled way, 
    converting them into actual lists. It also ensures proper formatting (i.e., normalizing 
    quotes to double quotes) before evaluating the string.

    Parameters:
        val (str or list): The input value, which could be a string representing a list 
                           or an actual list.

    Returns:
        list: The evaluated list if the input is a valid list-like string, or the 
              input list itself if it was already a list.
    
    Notes:
        This function uses `ast.literal_eval()` for safe evaluation, which only supports 
        Python literals and avoids executing arbitrary code.
    """
    
    # If the input value is already a list, return it as is
    if isinstance(val, list):
        return val
    else:
        # Remove leading and trailing whitespaces from the input string
        val = val.strip()
        
        # Normalize the string representation of a list to use double quotes for consistency
        if val.startswith('["') or val.startswith("['"):
            val = val.replace("['", '["').replace("']", '"]')  # Normalize to double quotes
        
        # Safely evaluate the string to convert it to a list
        return ast.literal_eval(val)
    

# Function to convert string representation of lists to actual lists for multiple columns
def convert_string_to_list(df, column_names):
    """
    Convert string representations of lists to actual Python lists for specified columns in a DataFrame.
    
    This function processes multiple columns of a DataFrame, where each specified column 
    contains string representations of lists, and converts those strings into actual lists 
    using the `safe_eval` function.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the columns to be processed.
        column_names (list of str): A list of column names whose values should be converted 
                                    from string representations to actual lists.

    Returns:
        pandas.DataFrame: The DataFrame with the specified columns converted from string 
                           representations to actual lists.
    
    Notes:
        This function relies on the `safe_eval()` function to safely evaluate the string 
        representations of lists and convert them into actual Python lists.
    """
    
    # Loop through each column name in the provided list
    for column_name in column_names:
        # Apply the safe_eval function to each value in the column to convert string representations of lists
        df[column_name] = df[column_name].apply(safe_eval)
    
    # Return the modified DataFrame
    return df


# Function to count occurrences of each string in the 'Actor Country of Origin' column
def count_occurrences_in_column(df, column_name):
    """
    Count the occurrences of individual values (strings) in a specified column of a DataFrame.
    
    This function takes a DataFrame and a column name as inputs. It iterates through the 
    values in the specified column, which may contain lists of strings. It flattens these 
    lists, processes the individual strings (splitting those containing multiple values, 
    such as country names separated by commas), and counts the occurrences of each unique 
    string. The function returns a Counter object with the counts of each unique string.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the column to be processed.
        column_name (str): The name of the column in the DataFrame to count occurrences in.
    
    Returns:
        collections.Counter: A Counter object with the count of occurrences for each 
                              unique value (string) in the specified column.
    
    Notes:
        - This function assumes that the values in the specified column are either lists of 
          strings or individual strings. 
        - If the list contains multiple values (such as country names), they will be 
          split by commas and counted separately.
        - Only non-empty, non-whitespace strings are counted.
    """
    
    # Initialize an empty list to store all the individual values
    all_values = []
    
    # Loop through each entry in the specified column
    for sublist in df[column_name]:
        if isinstance(sublist, list):  # Ensure the value is a list
            # Loop through each item in the list
            for item in sublist:
                if isinstance(item, str) and item.strip():  # Ensure the item is a non-empty string
                    # Split the string if it contains multiple values (e.g., "India, Pakistan")
                    countries = item.split(",")  # Split by commas
                    for country in countries:
                        # Clean up extra spaces or quotation marks from the string
                        country = country.strip().strip("'").strip('"')  
                        if country:  # Only add non-empty strings
                            all_values.append(country)
    
    # Return the count of occurrences of each unique string in the column
    return Counter(all_values)


def plot_gender_distribution(df, column='Actor Gender'):
    """
    Plots the distribution of a categorical column (e.g., 'Actor Gender') in a DataFrame.

    This function calculates the value counts of the specified column, calculates the percentage for each
    category, and then creates a bar plot displaying the frequency and percentage on top of each bar.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name for which the gender distribution is to be plotted (default is 'Actor Gender').

    Returns:
        None: The function directly shows the plot.
    """
    
    # Calculate the value counts for the specified column
    value_counts = df[column].value_counts()

    # Calculate the total count
    total_count = sum(value_counts)

    # Print the value counts
    print(value_counts)

    # Create a bar plot
    value_counts.plot(kind='bar')

    # Add value labels on top of each bar
    for i, value in enumerate(value_counts):
        percentage = (value / total_count) * 100
        plt.text(i, value + max(value_counts) * 0.01, f'{percentage:.1f}%', ha='center')
    
    # Customize the plot
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


def plot_height_distribution_by_gender(df, height_column='Actor Height', gender_column='Actor Gender'):
    """
    Creates a boxplot showing the distribution of actor height by gender, and adds median and standard deviation 
    values on the plot for each gender.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        height_column (str): The column name for actor heights (default is 'Actor Height').
        gender_column (str): The column name for actor gender (default is 'Actor Gender').

    Returns:
        None: The function directly shows the plot.
    """
    
    # Create a boxplot to show the distribution of Actor Height by Gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=gender_column, y=height_column, data=df)

    # Add median and standard deviation values on the plot for each gender
    for gender in df[gender_column].unique():
        median = df[df[gender_column] == gender][height_column].median()
        std_dev = df[df[gender_column] == gender][height_column].std()

        # Add the median and std dev on the plot
        plt.text(gender, median + 0.05, f'Median: {median:.2f}', horizontalalignment='center')
        plt.text(gender, median - 0.05, f'Std Dev: {std_dev:.2f}', horizontalalignment='center')

    # Customize the plot
    plt.title('Actor Height Distribution by Gender')
    plt.xlabel('Actor Gender')
    plt.ylabel('Actor Height')

    # Show the plot
    plt.show()



def plot_height_distribution(df, column_name):
    """
    Plots a bar chart showing the distribution of heights (or other numerical values) in a specified column.

    This function calculates the frequency of unique values in the specified column, sorts them by their 
    index (value), and creates a bar chart displaying these frequencies. It also calculates and displays 
    the percentage of each height in the total count.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column containing height (or other numerical) values.
    
    Returns:
        None: Displays a bar plot showing the distribution of heights.
    
    Notes:
        - The function assumes that the specified column contains numerical values.
        - A percentage label is added to each bar in the chart.
    """
    # Calculate value counts for the specified column, sorting by index (height values)
    value_counts_sorted = df[column_name].value_counts().sort_index()

    # Calculate the total count of the values
    total_count = sum(value_counts_sorted)

    # Create the bar plot
    value_counts_sorted.plot(kind='bar')

    # Add percentage labels on top of each bar
    for i, value in enumerate(value_counts_sorted):
        percentage = (value / total_count) * 100
        plt.text(i, value + max(value_counts_sorted) * 0.01, f'{percentage:.1f}%', ha='center')

    # Customize the plot (optional)
    plt.title(f'{column_name} Distribution')
    plt.xlabel(f'{column_name}')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


def process_and_plot(df, column_name, threshold=30):
    """
    Processes a specified column in the DataFrame, counts occurrences, 
    groups those below a threshold into an 'Other' category, and plots the results.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to process (either 'Actor Ethnicity' or 'Actor Country of Origin').
        threshold (int): The minimum count required for a category to be displayed separately.
                         Categories with counts below this will be grouped into 'Other'.
    
    Returns:
        None: Displays a count plot with percentages for the specified column.
    """
    
    def count_occurrences_in_column(df, column_name):
        """
        Helper function to count occurrences in a column, where values may be lists of strings.
        """
        all_values = []
        for sublist in df[column_name]:
            if isinstance(sublist, list):
                for item in sublist:
                    if isinstance(item, str) and item.strip():  # Check for non-empty string
                        values = item.split(",")  # Split by commas if multiple items
                        for value in values:
                            value = value.strip().strip("'").strip('"')  # Clean the string
                            if value:
                                all_values.append(value)
        return Counter(all_values)
    
    # Count occurrences of the specified column
    value_count = count_occurrences_in_column(df, column_name)

    # Convert Counter object to DataFrame
    df_value_count = pd.DataFrame(value_count.items(), columns=[column_name, 'Count']).set_index(column_name)

    # Print the value counts for reference
    print(f"{column_name} counts:")
    print(df_value_count)

    # Filter values below threshold
    other_values = df_value_count[df_value_count['Count'] >= threshold].index

    # Apply the logic to the column and group values below threshold as 'Other'
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].apply(
        lambda x: [value if value in other_values else 'Other' for value in x] 
        if isinstance(x, list) 
        else ('Other' if x not in other_values else x)
    )

    # Flatten the column for plotting (in case it's a list)
    df_copy_exploded = df_copy.explode(column_name)

    # Plot the count of values after regrouping
    plt.figure(figsize=(12, 10))
    sns.countplot(data=df_copy_exploded, y=column_name, 
                  order=df_copy_exploded[column_name].value_counts().index, color='orange')

    # Add percentages to the plot
    total = len(df_copy_exploded)
    ax = plt.gca()
    for p in ax.patches:
        percentage = f'{100 * p.get_width() / total:.1f}%'
        plt.text(p.get_width() + 5, p.get_y() + p.get_height() / 2, percentage, ha='left', va='center')

    # Customize the plot
    plt.title(f'Count (percentage) of actors by {column_name} with Other regrouping for values below {threshold} counts')
    plt.xlabel('Count')
    plt.ylabel(column_name)
    plt.show()


def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for association between two categorical variables.
    
    Cramér's V is a measure of association between two categorical variables, 
    ranging from 0 (no association) to 1 (perfect association). It is based on the 
    Chi-squared statistic and is corrected for small sample sizes.
    
    Parameters:
        x (pandas.Series): A categorical variable (e.g., a column from a DataFrame).
        y (pandas.Series): A categorical variable (e.g., another column from a DataFrame).
    
    Returns:
        float: The Cramér's V statistic, representing the strength of the association between x and y.
    """
    
    # Step 1: Create a contingency table from the two categorical variables x and y
    contingency_table = pd.crosstab(x, y)

    # Step 2: Calculate the Chi-squared statistic from the contingency table
    chi2 = chi2_contingency(contingency_table)[0]

    # Step 3: Get the total number of observations (the sum of all entries in the contingency table)
    n = contingency_table.sum().sum()

    # Step 4: Calculate the phi-squared statistic, which is the Chi-squared statistic divided by the total number of observations
    phi2 = chi2 / n

    # Step 5: Get the number of rows (r) and columns (k) in the contingency table
    r, k = contingency_table.shape

    # Step 6: Correct the phi-squared statistic for cases where the sample size is small
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))

    # Step 7: Correct the number of rows (r) for the calculation of Cramér's V
    rcorr = r - (r - 1) ** 2 / (n - 1)

    # Step 8: Correct the number of columns (k) for the calculation of Cramér's V
    kcorr = k - (k - 1) ** 2 / (n - 1)

    # Step 9: Return the square root of the corrected phi-squared statistic divided by the smaller of the two corrections
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def compute_cramers_v_heatmap(df, columns, cramers_v_func):
    """
    Computes and visualizes the Cramér's V correlation matrix for specified columns in a DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list): A list of column names to consider for the correlation.
        cramers_v_func (function): The Cramér's V calculation function (e.g., 'cramers_v').

    Returns:
        None: Displays a heatmap of the Cramér's V matrix.
    """
    # Concatenate the specified columns
    df_attributes = df[columns]

    # Calculate Cramér's V for all pairs of categorical variables
    categories = df_attributes.columns
    cramers_v_matrix = pd.DataFrame(index=categories, columns=categories)

    for cat1 in categories:
        for cat2 in categories:
            if cat1 == cat2:
                cramers_v_matrix.loc[cat1, cat2] = 1.0  # Perfect correlation with itself
            else:
                cramers_v_matrix.loc[cat1, cat2] = cramers_v_func(df_attributes[cat1], df_attributes[cat2])

    # Convert to float for proper visualization
    cramers_v_matrix = cramers_v_matrix.astype(float)

    # Plotting the Cramér's V heatmap
    plt.figure(figsize=(8, 6))  # Adjust the figure size as necessary
    sns.heatmap(cramers_v_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    # Show the plot
    plt.title("Cramér's V Correlation Matrix Heatmap")
    plt.show()
