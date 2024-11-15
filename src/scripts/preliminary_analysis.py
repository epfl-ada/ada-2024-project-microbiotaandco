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

