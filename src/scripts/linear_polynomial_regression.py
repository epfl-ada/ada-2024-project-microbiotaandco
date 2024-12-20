# Linear and Polynomial Regression implementation for time analysis of diversity scores
# Important: Ensure 'diversity_score_functions.py' is in the same directory 


#Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import os

#Data --------------------------------------------------------------------
#mean gender score
from plot_diversity_country_time import Diversity_movie_metadata
df_yearly = Diversity_movie_metadata.groupby('Movie Release Date')['gender_score'].mean().reset_index()

#define variables
X = df_yearly['Movie Release Date'].values.reshape(-1, 1)  
y = df_yearly['gender_score'].values  


#Linear Regression fit--------------------------------------------------------------------

#### 1 #####First plots for LR model

#list of diversity scores and their respective labels for plotting
diversity_scores = ['gender_score', 'age_score', 'height_score', 'ethnicity_score', 'Foreign Actor Proportion', 'diversity_score']
score_labels = ['Gender', 'Age', 'Height', 'Ethnicity', 'Foreign Actors', 'Diversity Score']
#choice of the colors based on old movies palette
colors = ['thistle', 'palegoldenrod', 'lightsalmon', 'mediumseagreen', 'lightblue', 'indianred'] 

# subplots: one row, number of columns equal to the number of diversity scores
fig, axes = plt.subplots(1, len(diversity_scores), figsize=(30, 6))

# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X = df_yearly['Movie Release Date'].values.reshape(-1, 1)  
    y = df_yearly[score].values  

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the values for the regression line
    y_pred = model.predict(X)

    # Plot the original data and regression line
    ax = axes[i]  # Access the i-th subplot axis
    ax.plot(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    ax.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Linear Regression line')


    # Fill the area under the mean gender score
    ax.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], alpha=0.5, label=f'Mean {score_labels[i]} score area')

    # Add R-squared and regression equation text below the plot
    r_squared = model.score(X, y)
    equation_text = f'{score_labels[i]} Score = {model.coef_[0]:.4f} * Year  {model.intercept_:.2f}'
    r_squared_text = f'R² = {r_squared:.4f}'

    # Place the regression equation and R-squared value outside the plot
    fig.text(0.05 + i * 0.2, -0.03, equation_text, ha='left', va='top', fontsize=16, color='red')
    fig.text(0.05 + i * 0.2, -0.08, r_squared_text, ha='left', va='top', fontsize=16, color='black')

    # Set labels and title for each subplot
    ax.set_xlabel('Movie release year', fontsize=14)
    ax.set_ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    ax.set_title(f'Time evolution of the mean {score_labels[i]} score', fontsize=14)
    ax.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')

    ax.legend()

# Adjust layout to avoid overlap and add extra space for equation and R-squared
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()

########## 2 ########## Second plots for LR model

# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    #group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    X_centered = df_yearly['Movie Release Date']  
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  

    #fit
    model = LinearRegression()
    model.fit(X, y)

    #predict
    y_pred = model.predict(X)

    #new figure for each diversity score
    plt.figure(figsize=(10, 6))

    # Plot the original data and regression line
    plt.plot(df_yearly['Movie Release Date'], df_yearly[score],  color=colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    plt.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Linear Regression line')
    plt.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color=colors[i], alpha=0.5, label='Mean Score Area')

    # Add R-squared and regression equation text below the plot
    r_squared = model.score(X, y)
    equation_text = f'{score_labels[i]} Score = {model.coef_[0]:.4f} * Year + {model.intercept_:.2f}'
    r_squared_text = f'R² = {r_squared:.4f}'

    # Place the regression equation and R-squared value outside the plot
    plt.figtext(0.03, -0.02, equation_text, ha='left', va='top', fontsize=16, color='red')
    plt.figtext(0.03, -0.08, r_squared_text, ha='left', va='top', fontsize=16, color='black')

    # Set labels and title for the plot
    plt.xlabel('Movie release year', fontsize=14)
    plt.ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    plt.title(f'Time evolution of the mean {score_labels[i]} score.', fontsize=14)
    plt.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')
    plt.legend()

    # Adjust layout to avoid overlap and add extra space for equation and R-squared
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


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
# Create the bar plot
plt.figure(figsize=(10, 6))

# Make a barplot of the coefficients with on the x axis the dfs and on the y the coefficients
labels = [result[0] for result in results]
coefficients = [result[1] for result in results]

bars = plt.bar(labels, coefficients, color='darkgrey', alpha=0.7)

# Add the exact value of the regression slope for each coefficient above each bar
for bar, (label, coef, _) in zip(bars, results):
    # Positioning the text at the top of each bar with some padding
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




# Create a subplot grid: one row, number of columns equal to the number of diversity scores
fig, axes = plt.subplots( 1, len(diversity_scores)-1, figsize=(30, 8))

# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores[:-1] ):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X_centered = df_yearly['Movie Release Date']  # Centering around 1895
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  


    degree = 3
    # polynomial feature creation with defined degree
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the values for the regression line
    y_pred = model.predict(X_poly)

    # Plot the original data and regression line
    ax = axes[i]  # Access the i-th subplot axis
    ax.plot(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    ax.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Polynomial Regression line')
    ax.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], alpha=0.5, label='Mean Score Area')

    # Add R-squared and regression equation text below the plot
    #add regression equation
    coeffs = model.coef_
    intercept = model.intercept_
    r_squared = model.score(X_poly, y)
    equation_text = f"y = {intercept:.3f} + " + " + ".join([f"{coeffs[i]:.3f}x^{i}" for i in range(1, len(coeffs))])
    r_squared_text = f'R² = {r_squared:.4f}'

    # Place the regression equation and R-squared value outside the plot
    fig.text(0.01 + i * 0.2, -0.05, equation_text, ha='left', va='top', fontsize=14, color='red')
    fig.text(0.01 + i * 0.2, -0.08, r_squared_text, ha='left', va='top', fontsize=14, color='black')
    fig.text(0.01 + i * 0.2, -0.02, f'Polynomial degree = {degree}', ha='left', va='top', fontsize=14, color='black')


    plt.title('Critical and Inflection Points of Polynomial Regression')
    # Set labels and title for each subplot
    ax.set_xlabel('Movie release year', fontsize=14)
    ax.set_ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    ax.set_title(f'Time evolution of the mean {score_labels[i]} score.', fontsize=14)
    ax.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')
    ax.legend()

    

    #show explicitly the model's coefficients and intercept
    print(f'Polynomial regression coefficients: {coeffs}')
    print(f'Polynomial regression intercept: {intercept}')

# Adjust layout to avoid overlap and add extra space for equation and R-squared
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()



# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X_centered = df_yearly['Movie Release Date']  # Centering around 1895
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  

    degree = 3
    # polynomial feature creation with defined degree
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the values for the regression line
    y_pred = model.predict(X_poly)

    # Create a new figure for each diversity score
    plt.figure(figsize=(10, 6))

    # Plot the original data and regression line
    plt.plot(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    plt.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Polynomial Regression line')
    plt.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color= colors[i], alpha=0.5, label='Mean Score Area')

    # Add R-squared and regression equation text below the plot
    coeffs = model.coef_
    intercept = model.intercept_
    r_squared = model.score(X_poly, y)
    equation_text = f"y = {intercept:.3f} + " + " + ".join([f"{coeffs[i]:.3f}x^{i}" for i in range(1, len(coeffs))])
    r_squared_text = f'R² = {r_squared:.4f}'

    # Place the regression equation and R-squared value outside the plot
    plt.figtext(0.01 , -0.05, equation_text, ha='left', va='top', fontsize=12, color='red')
    plt.figtext(0.01 , -0.08, r_squared_text, ha='left', va='top', fontsize=12, color='black')
    plt.figtext(0.01 , -0.02, f'Polynomial degree = {degree}', ha='left', va='top', fontsize=12, color='black')


    # Set labels and title for the plot
    plt.xlabel('Movie release year', fontsize=14)
    plt.ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    plt.title(f'Time evolution of the mean {score_labels[i]} score.', fontsize=14)
    plt.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')
    plt.legend()

    # Show explicitly the model's coefficients and intercept
    print(f'Polynomial regression coefficients for {score_labels[i]}: {coeffs}')
    print(f'Polynomial regression intercept for {score_labels[i]}. : {intercept}')

    # Adjust layout to avoid overlap and add extra space for equation and R-squared
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()





# Create a subplot grid: one row, number of columns equal to the number of diversity scores
fig, axes = plt.subplots( 1, len(diversity_scores), figsize=(30, 8))

# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X_centered = df_yearly['Movie Release Date']  # Centering around 1895
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  

    # Polynomial feature creation with defined degree
    degree = 3
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the values for the regression line
    y_pred = model.predict(X_poly)

    # Get coefficients and intercept for the polynomial regression
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

    # Plot the original data and regression line
    ax = axes[i]  # Access the i-th subplot axis
    ax.plot(df_yearly['Movie Release Date'], df_yearly[score], color=colors[i], linestyle='-', linewidth=2, markersize=5, label=f'Mean {score_labels[i]} score')
    ax.plot(df_yearly['Movie Release Date'], y_pred, color='red', linestyle='--', linewidth=2, label='Polynomial Regression line')
    ax.fill_between(df_yearly['Movie Release Date'], df_yearly[score], color=colors[i], alpha=0.5, label='Mean Score Area')

    # Scatter the critical and inflection points
    ax.scatter(critical_points, p(critical_points), color='darkred', s=100, zorder=5, label='Critical points')
    ax.scatter(inflection_points, p(inflection_points), color='steelblue', s=100, zorder=5, label='Inflection points')

    # Add R-squared and regression equation text below the plot
    r_squared = model.score(X_poly, y)
    equation_text = f"y = {intercept:.3f} + " + " + ".join([f"{coeffs[i]:.3f}x^{i}" for i in range(1, len(coeffs))])
    r_squared_text = f'R² = {r_squared:.4f}'

    # Place the regression equation and R-squared value outside the plot
    fig.text(0.01 + i * 0.2, -0.05, equation_text, ha='left', va='top', fontsize=14, color='red')
    fig.text(0.01 + i * 0.2, -0.08, r_squared_text, ha='left', va='top', fontsize=14, color='black')
    fig.text(0.01 + i * 0.2, -0.02, f'Polynomial degree = {degree}', ha='left', va='top', fontsize=14, color='black')


    # Set labels and title for each subplot
    ax.set_xlabel('Movie release year', fontsize=14)
    ax.set_ylabel(f'Mean {score_labels[i]} score', fontsize=14)
    ax.set_title(f'Time evolution of the mean {score_labels[i]} score.', fontsize=14)
    ax.grid(axis='x', linestyle='-', linewidth=0.5, color='grey')
    ax.legend()

    # Show explicitly the model's coefficients and intercept
    print(f'Polynomial regression coefficients: {coeffs}')
    print(f'Polynomial regression intercept: {intercept}')

# Adjust layout to avoid overlap and add extra space for equation and R-squared
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()




# Loop over each diversity score and plot
for i, score in enumerate(diversity_scores):
    data_copy = Diversity_movie_metadata.copy()
    # Group by movie release year and calculate the mean diversity score
    df_yearly = data_copy.groupby('Movie Release Date')[diversity_scores[i]].mean().reset_index()

    # Define the variables for regression
    X_centered = df_yearly['Movie Release Date']  # Centering around 1895
    X = X_centered.values.reshape(-1, 1)  
    y = df_yearly[score].values  

    # Polynomial feature creation with defined degree
    degree = 3
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the values for the regression line
    y_pred = model.predict(X_poly)

    # Get coefficients and intercept for the polynomial regression
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

    # Show explicitly the model's coefficients and intercept
    print(f'Polynomial regression coefficients for {score_labels[i]}: {coeffs}')
    print(f'Polynomial regression intercept for {score_labels[i]}: {intercept}')

    # Adjust layout to avoid overlap and add extra space for equation and R-squared
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


