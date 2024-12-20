
import pandas as pd
import ast
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd




Diversity_movie_metadata=pd.read_csv('Diversity_movie_metadata.csv')



# Load your dataset
df = Diversity_movie_metadata.copy()

# Convert string representation of lists to actual lists if needed
df['Movie Country'] = df['Movie Country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode the 'Movie Country' column to create one row per country
df = df.explode('Movie Country')

# Strip whitespace from country names
df['Movie Country'] = df['Movie Country'].str.strip()

# Categorize movies by release year
def categorize_year(year):
    if year < 1975:
        return 'Old Movies (Before 1975)'
    elif 1975 <= year <= 1990:
        return 'Movies (1975-1990)'
    else:
        return 'New Movies (After 1990)'

df['Movie Category'] = df['Movie Release Date'].apply(categorize_year)

# Group by country and category, then compute the average diversity score
country_year_scores = (
    df.groupby(['Movie Country', 'Movie Category'])['Foreign Actor Proportion']
    .mean()
    .reset_index()
    .rename(columns={'Movie Country': 'country', 'Movie Category': 'category', 'diversity_score': 'Foreign Actor Proportion'})
)

# Ensure country names are consistent
country_year_scores['country'] = country_year_scores['country'].str.strip()

# Display a sample of the resulting DataFrame
country_year_scores.sample(10)


#tryFOREIGN


# Load GeoJSON world boundaries
world = gpd.read_file('realworld.json')
world['admin'] = world['admin'].str.strip()  # Remove spaces from country names

# Ensure country names in your dataset match the GeoJSON
country_year_scores['country'] = country_year_scores['country'].str.strip()

# Add your "low movie" lists for each category
low_movie_countries = {
    'Old Movies (Before 1975)': ['Canada'],
    'Movies (1975-1990)': ['Austria', 'China', 'South Korea', 'Spain'],
    'New Movies (After 1990)': ['Egypt', 'Peru', 'Thailand', 'Turkey']
}

# Calculate global min and max for the color scale
global_min = country_year_scores['Foreign Actor Proportion'].min()
global_max = country_year_scores['Foreign Actor Proportion'].max()
print(f"Global Min: {global_min}, Global Max: {global_max}")  # Debug

# Loop through each category and save the plots
unique_categories = sorted(country_year_scores['category'].unique())
print(f"Categories in dataset: {unique_categories}")  # Debug

for category in unique_categories:
    # Filter the data for the current category
    category_data = country_year_scores[country_year_scores['category'] == category]
    print(f"Processing for category '{category}' with {len(category_data)} records.")  # Debug
    
    # Merge GeoJSON with the category's data
    geo_data = world.merge(category_data, left_on='admin', right_on='country', how='left')
    
    # Identify countries with low movie counts for the current category
    low_movie_list = low_movie_countries.get(category, [])
    geo_data['low_movie'] = geo_data['admin'].isin(low_movie_list)
    
    # Plot the merged data
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    geo_data.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot boundaries

    # Highlight Canada in blue
    geo_data.boundary[geo_data['admin'] == 'Canada'].plot(ax=ax, linewidth=2, edgecolor='red', zorder=1)
    # Plot the choropleth map
    geo_data.plot(
        column='Foreign Actor Proportion',
        ax=ax,
        legend=True,
        cmap='Blues',
        vmin=global_min,  # Set global min for color scale
        vmax=global_max,  # Set global max for color scale
        missing_kwds={"color": "lightgrey", "label": "No Data"},
        legend_kwds={
            "shrink": 0.5,  # Shrink the color scale legend
            "label": "Foreign Actor Proportion",  # Add a label to the legend
            "orientation": "vertical"  # Ensure the legend is vertical
        }
    )
    
    # Overlay stripes for low-movie countries
    geo_data[geo_data['low_movie']].boundary.plot(
        ax=ax,
        linewidth=1,
        linestyle='--',  # Dashed lines for stripes
        edgecolor='red',  # Use a distinct color for stripes
        alpha=0.7
    )
    
    # Add a title
    ax.set_title(f"{category} - Foreign Actor Proportion by Country", fontsize=16)
    ax.axis('off')  # Turn off the axis for a cleaner map
    
    # Save the plot as a JPEG file
    output_filename = f"choropleth_{category.replace(' ', '_').lower()}.jpg"
    plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight')
    print(f"Saved plot for category '{category}' as '{output_filename}'.")
    plt.close(fig)  # Close the figure to free memory




# Load your dataset
df1 = Diversity_movie_metadata.copy()

# Convert string representation of lists to actual lists if needed
df1['Movie Country'] = df1['Movie Country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode the 'Movie Country' column to create one row per country
df1 = df1.explode('Movie Country')

# Strip whitespace from country names
df1['Movie Country'] = df1['Movie Country'].str.strip()

# Categorize movies by release year
def categorize_year(year):
    if year < 1975:
        return 'Old Movies (Before 1975)'
    elif 1975 <= year <= 1990:
        return 'Movies (1975-1990)'
    else:
        return 'New Movies (After 1990)'

df1['Movie Category'] = df1['Movie Release Date'].apply(categorize_year)

# Group by country and category, then compute the average diversity score
country_year_scores1 = (
    df1.groupby(['Movie Country', 'Movie Category'])['gender_score']
    .mean()
    .reset_index()
    .rename(columns={'Movie Country': 'country', 'Movie Category': 'category', 'diversity_score': 'gender_score'})
)

# Ensure country names are consistent
country_year_scores1['country'] = country_year_scores1['country'].str.strip()

# Display a sample of the resulting DataFrame
country_year_scores1.sample(10)

#trygender

# Load GeoJSON world boundaries
world = gpd.read_file('realworld.json')
world['admin'] = world['admin'].str.strip()  # Remove spaces from country names

# Ensure country names in your dataset match the GeoJSON
country_year_scores1['country'] = country_year_scores1['country'].str.strip()

# Add your "low movie" lists for each category
low_movie_countries = {
    'Old Movies (Before 1975)': ['Canada'],
    'Movies (1975-1990)': ['Austria', 'China', 'South Korea', 'Spain'],
    'New Movies (After 1990)': ['Egypt', 'Peru', 'Thailand', 'Turkey']
}

# Calculate global min and max for the color scale
global_min = country_year_scores1['gender_score'].min()
global_max = country_year_scores1['gender_score'].max()
print(f"Global Min: {global_min}, Global Max: {global_max}")  # Debug

# Loop through each category and save the plots
unique_categories = sorted(country_year_scores1['category'].unique())
print(f"Categories in dataset: {unique_categories}")  # Debug

for category in unique_categories:
    # Filter the data for the current category
    category_data = country_year_scores1[country_year_scores1['category'] == category]
    print(f"Processing for category '{category}' with {len(category_data)} records.")  # Debug
    
    # Merge GeoJSON with the category's data
    geo_data = world.merge(category_data, left_on='admin', right_on='country', how='left')
    
    # Identify countries with low movie counts for the current category
    low_movie_list = low_movie_countries.get(category, [])
    geo_data['low_movie'] = geo_data['admin'].isin(low_movie_list)
    
    # Plot the merged data
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    geo_data.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot boundaries

    # Highlight Canada in blue
   
    # Plot the choropleth map
    geo_data.plot(
        column='gender_score',
        ax=ax,
        legend=True,
        cmap='Purples',
        vmin=global_min,  # Set global min for color scale
        vmax=global_max,  # Set global max for color scale
        missing_kwds={"color": "lightgrey", "label": "No Data"},
        legend_kwds={
            "shrink": 0.5,  # Shrink the color scale legend
            "label": "gender_score",  # Add a label to the legend
            "orientation": "vertical"  # Ensure the legend is vertical
        }
    )
    
    # Overlay stripes for low-movie countries
    geo_data[geo_data['low_movie']].boundary.plot(
        ax=ax,
        linewidth=1,
        linestyle='--',  # Dashed lines for stripes
        edgecolor='red',  # Use a distinct color for stripes
        alpha=0.7
    )
    
    # Add a title
    ax.set_title(f"{category} - gender score Proportion by Country", fontsize=16)
    ax.axis('off')  # Turn off the axis for a cleaner map
    
    # Save the plot as a JPEG file
    output_filename = f"choropleth_{category.replace(' ', '_').lower()}.jpg"
    plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight')
    print(f"Saved plot for category '{category}' as '{output_filename}'.")
    plt.close(fig)  # Close the figure to free memory





# Load your dataset
df2 = Diversity_movie_metadata.copy()

# Convert string representation of lists to actual lists if needed
df2['Movie Country'] = df2['Movie Country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode the 'Movie Country' column to create one row per country
df2 = df2.explode('Movie Country')

# Strip whitespace from country names
df2['Movie Country'] = df2['Movie Country'].str.strip()

# Categorize movies by release year
def categorize_year(year):
    if year < 1975:
        return 'Old Movies (Before 1975)'
    elif 1975 <= year <= 1990:
        return 'Movies (1975-1990)'
    else:
        return 'New Movies (After 1990)'

df2['Movie Category'] = df2['Movie Release Date'].apply(categorize_year)

# Group by country and category, then compute the average diversity score
country_year_scores2 = (
    df2.groupby(['Movie Country', 'Movie Category'])['ethnicity_score']
    .mean()
    .reset_index()
    .rename(columns={'Movie Country': 'country', 'Movie Category': 'category', 'diversity_score': 'ethnicity_score'})
)

# Ensure country names are consistent
country_year_scores2['country'] = country_year_scores2['country'].str.strip()

# Display a sample of the resulting DataFrame
country_year_scores2.sample(10)


#tryethnicity


# Load GeoJSON world boundaries
world = gpd.read_file('realworld.json')
world['admin'] = world['admin'].str.strip()  # Remove spaces from country names

# Ensure country names in your dataset match the GeoJSON
country_year_scores2['country'] = country_year_scores2['country'].str.strip()

# Add your "low movie" lists for each category
low_movie_countries = {
    'Old Movies (Before 1975)': ['Canada'],
    'Movies (1975-1990)': ['Austria', 'China', 'South Korea', 'Spain'],
    'New Movies (After 1990)': ['Egypt', 'Peru', 'Thailand', 'Turkey']
}

# Calculate global min and max for the color scale
global_min = country_year_scores2['ethnicity_score'].min()
global_max = country_year_scores2['ethnicity_score'].max()
print(f"Global Min: {global_min}, Global Max: {global_max}")  # Debug

# Loop through each category and save the plots
unique_categories = sorted(country_year_scores2['category'].unique())
print(f"Categories in dataset: {unique_categories}")  # Debug

for category in unique_categories:
    # Filter the data for the current category
    category_data = country_year_scores2[country_year_scores2['category'] == category]
    print(f"Processing for category '{category}' with {len(category_data)} records.")  # Debug
    
    # Merge GeoJSON with the category's data
    geo_data = world.merge(category_data, left_on='admin', right_on='country', how='left')
    
    # Identify countries with low movie counts for the current category
    low_movie_list = low_movie_countries.get(category, [])
    geo_data['low_movie'] = geo_data['admin'].isin(low_movie_list)
    
    # Plot the merged data
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    geo_data.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot boundaries

    # Plot the choropleth map
    geo_data.plot(
        column='ethnicity_score',
        ax=ax,
        legend=True,
        cmap='Greens',
        vmin=global_min,  # Set global min for color scale
        vmax=global_max,  # Set global max for color scale
        missing_kwds={"color": "lightgrey", "label": "No Data"},
        legend_kwds={
            "shrink": 0.5,  # Shrink the color scale legend
            "label": "ethnicity_score",  # Add a label to the legend
            "orientation": "vertical"  # Ensure the legend is vertical
        }
    )
    
    # Overlay stripes for low-movie countries
    geo_data[geo_data['low_movie']].boundary.plot(
        ax=ax,
        linewidth=1,
        linestyle='--',  # Dashed lines for stripes
        edgecolor='red',  # Use a distinct color for stripes
        alpha=0.7
    )
    
    # Add a title
    ax.set_title(f"{category} - ethnicity score Proportion by Country", fontsize=16)
    ax.axis('off')  # Turn off the axis for a cleaner map
    
    # Save the plot as a JPEG file
    output_filename = f"choropleth_{category.replace(' ', '_').lower()}.jpg"
    plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight')
    print(f"Saved plot for category '{category}' as '{output_filename}'.")
    plt.close(fig)  # Close the figure to free memory


# Load your dataset
df3 = Diversity_movie_metadata.copy()

# Convert string representation of lists to actual lists if needed
df3['Movie Country'] = df3['Movie Country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode the 'Movie Country' column to create one row per country
df3 = df3.explode('Movie Country')

# Strip whitespace from country names
df3['Movie Country'] = df3['Movie Country'].str.strip()

# Categorize movies by release year
def categorize_year(year):
    if year < 1975:
        return 'Old Movies (Before 1975)'
    elif 1975 <= year <= 1990:
        return 'Movies (1975-1990)'
    else:
        return 'New Movies (After 1990)'

df3['Movie Category'] = df3['Movie Release Date'].apply(categorize_year)

# Group by country and category, then compute the average diversity score
country_year_scores3 = (
    df3.groupby(['Movie Country', 'Movie Category'])['diversity_score']
    .mean()
    .reset_index()
    .rename(columns={'Movie Country': 'country', 'Movie Category': 'category', 'diversity_score': 'diversity_score'})
)

# Ensure country names are consistent
country_year_scores3['country'] = country_year_scores3['country'].str.strip()

# Display a sample of the resulting DataFrame
country_year_scores3.sample(10)

   
#trydiversity


# Load GeoJSON world boundaries
world = gpd.read_file('realworld.json')
world['admin'] = world['admin'].str.strip()  # Remove spaces from country names

# Ensure country names in your dataset match the GeoJSON
country_year_scores3['country'] = country_year_scores3['country'].str.strip()

# Add your "low movie" lists for each category
low_movie_countries = {
    'Old Movies (Before 1975)': ['Canada'],
    'Movies (1975-1990)': ['Austria', 'China', 'South Korea', 'Spain'],
    'New Movies (After 1990)': ['Egypt', 'Peru', 'Thailand', 'Turkey']
}

# Calculate global min and max for the color scale
global_min = country_year_scores3['diversity_score'].min()
global_max = country_year_scores3['diversity_score'].max()
print(f"Global Min: {global_min}, Global Max: {global_max}")  # Debug

# Loop through each category and save the plots
unique_categories = sorted(country_year_scores3['category'].unique())
print(f"Categories in dataset: {unique_categories}")  # Debug

for category in unique_categories:
    # Filter the data for the current category
    category_data = country_year_scores3[country_year_scores3['category'] == category]
    print(f"Processing for category '{category}' with {len(category_data)} records.")  # Debug
    
    # Merge GeoJSON with the category's data
    geo_data = world.merge(category_data, left_on='admin', right_on='country', how='left')
    
    # Identify countries with low movie counts for the current category
    low_movie_list = low_movie_countries.get(category, [])
    geo_data['low_movie'] = geo_data['admin'].isin(low_movie_list)
    
    # Plot the merged data
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    geo_data.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot boundaries
    
    # Plot the choropleth map
    geo_data.plot(
        column='diversity_score',
        ax=ax,
        legend=True,
        cmap='Reds',
        vmin=global_min,  # Set global min for color scale
        vmax=global_max,  # Set global max for color scale
        missing_kwds={"color": "lightgrey", "label": "No Data"},
        legend_kwds={
            "shrink": 0.5,  # Shrink the color scale legend
            "label": "diversity_score",  # Add a label to the legend
            "orientation": "vertical"  # Ensure the legend is vertical
        }
    )
    
    # Overlay stripes for low-movie countries
    geo_data[geo_data['low_movie']].boundary.plot(
        ax=ax,
        linewidth=1,
        linestyle='--',  # Dashed lines for stripes
        edgecolor='blue',  # Use a distinct color for stripes
        alpha=0.7
    )
    
    # Add a title
    ax.set_title(f"{category} - diversity score Proportion by Country", fontsize=16)
    ax.axis('off')  # Turn off the axis for a cleaner map
    
    # Save the plot as a JPEG file
    output_filename = f"choropleth_{category.replace(' ', '_').lower()}.jpg"
    plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight')
    print(f"Saved plot for category '{category}' as '{output_filename}'.")
    plt.close(fig)  # Close the figure to free memory


# Load your dataset
df4 = Diversity_movie_metadata.copy()

# Convert string representation of lists to actual lists if needed
df4['Movie Country'] = df4['Movie Country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode the 'Movie Country' column to create one row per country
df4 = df4.explode('Movie Country')

# Strip whitespace from country names
df4['Movie Country'] = df4['Movie Country'].str.strip()

# Categorize movies by release year
def categorize_year(year):
    if year < 1975:
        return 'Old Movies (Before 1975)'
    elif 1975 <= year <= 1990:
        return 'Movies (1975-1990)'
    else:
        return 'New Movies (After 1990)'

df4['Movie Category'] = df4['Movie Release Date'].apply(categorize_year)

# Group by country and category, then compute the average diversity score
country_year_scores4 = (
    df4.groupby(['Movie Country', 'Movie Category'])['height_score']
    .mean()
    .reset_index()
    .rename(columns={'Movie Country': 'country', 'Movie Category': 'category', 'diversity_score': 'height_score'})
)

# Ensure country names are consistent
country_year_scores4['country'] = country_year_scores4['country'].str.strip()

# Display a sample of the resulting DataFrame
country_year_scores4.sample(10)

#tryheight

# Load GeoJSON world boundaries
world = gpd.read_file('realworld.json')
world['admin'] = world['admin'].str.strip()  # Remove spaces from country names

# Ensure country names in your dataset match the GeoJSON
country_year_scores4['country'] = country_year_scores4['country'].str.strip()

# Add your "low movie" lists for each category
low_movie_countries = {
    'Old Movies (Before 1975)': ['Canada'],
    'Movies (1975-1990)': ['Austria', 'China', 'South Korea', 'Spain'],
    'New Movies (After 1990)': ['Egypt', 'Peru', 'Thailand', 'Turkey']
}

# Calculate global min and max for the color scale
global_min = country_year_scores4['height_score'].min()
global_max = country_year_scores4['height_score'].max()
print(f"Global Min: {global_min}, Global Max: {global_max}")  # Debug

# Loop through each category and save the plots
unique_categories = sorted(country_year_scores4['category'].unique())
print(f"Categories in dataset: {unique_categories}")  # Debug

for category in unique_categories:
    # Filter the data for the current category
    category_data = country_year_scores4[country_year_scores4['category'] == category]
    print(f"Processing for category '{category}' with {len(category_data)} records.")  # Debug
    
    # Merge GeoJSON with the category's data
    geo_data = world.merge(category_data, left_on='admin', right_on='country', how='left')
    
    # Identify countries with low movie counts for the current category
    low_movie_list = low_movie_countries.get(category, [])
    geo_data['low_movie'] = geo_data['admin'].isin(low_movie_list)
    
    # Plot the merged data
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    geo_data.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot boundaries
    
    # Plot the choropleth map
    geo_data.plot(
        column='height_score',
        ax=ax,
        legend=True,
        cmap='Oranges',
        vmin=global_min,  # Set global min for color scale
        vmax=global_max,  # Set global max for color scale
        missing_kwds={"color": "lightgrey", "label": "No Data"},
        legend_kwds={
            "shrink": 0.5,  # Shrink the color scale legend
            "label": "height_score",  # Add a label to the legend
            "orientation": "vertical"  # Ensure the legend is vertical
        }
    )
    
    # Overlay stripes for low-movie countries
    geo_data[geo_data['low_movie']].boundary.plot(
        ax=ax,
        linewidth=1,
        linestyle='--',  # Dashed lines for stripes
        edgecolor='green',  # Use a distinct color for stripes
        alpha=0.7
    )
    
    # Add a title
    ax.set_title(f"{category} - height score Proportion by Country", fontsize=16)
    ax.axis('off')  # Turn off the axis for a cleaner map
    
    # Save the plot as a JPEG file
    output_filename = f"choropleth_{category.replace(' ', '_').lower()}.jpg"
    plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight')
    print(f"Saved plot for category '{category}' as '{output_filename}'.")
    plt.close(fig)  # Close the figure to free memory





     # Load your dataset
df5 = Diversity_movie_metadata.copy()

# Convert string representation of lists to actual lists if needed
df5['Movie Country'] = df5['Movie Country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Explode the 'Movie Country' column to create one row per country
df5 = df5.explode('Movie Country')

# Strip whitespace from country names
df5['Movie Country'] = df5['Movie Country'].str.strip()

# Categorize movies by release year
def categorize_year(year):
     if year < 1975:
        return 'Old Movies (Before 1975)'
     elif 1975 <= year <= 1990:
        return 'Movies (1975-1990)'
     else:
        return 'New Movies (After 1990)'

df5['Movie Category'] = df5['Movie Release Date'].apply(categorize_year)

# Group by country and category, then compute the average diversity score
country_year_scores5 = (
    df5.groupby(['Movie Country', 'Movie Category'])['age_score']
    .mean()
    .reset_index()
    .rename(columns={'Movie Country': 'country', 'Movie Category': 'category', 'diversity_score': 'age_score'})
)

# Ensure country names are consistent
country_year_scores5['country'] = country_year_scores5['country'].str.strip()

# Display a sample of the resulting DataFrame
country_year_scores5.sample(10)





#tryage
# Load GeoJSON world boundaries
world = gpd.read_file('realworld.json')
world['admin'] = world['admin'].str.strip()  # Remove spaces from country names

# Ensure country names in your dataset match the GeoJSON
country_year_scores5['country'] = country_year_scores5['country'].str.strip()

# Add your "low movie" lists for each category
low_movie_countries = {
    'Old Movies (Before 1975)': ['Canada'],
    'Movies (1975-1990)': ['Austria', 'China', 'South Korea', 'Spain'],
    'New Movies (After 1990)': ['Egypt', 'Peru', 'Thailand', 'Turkey']
}

# Calculate global min and max for the color scale
global_min = country_year_scores5['age_score'].min()
global_max = country_year_scores5['age_score'].max()
print(f"Global Min: {global_min}, Global Max: {global_max}")  # Debug

# Loop through each category and save the plots
unique_categories = sorted(country_year_scores5['category'].unique())
print(f"Categories in dataset: {unique_categories}")  # Debug

for category in unique_categories:
        # Filter the data for the current category
        category_data = country_year_scores5[country_year_scores5['category'] == category]
        print(f"Processing for category '{category}' with {len(category_data)} records.")  # Debug
        
        # Merge GeoJSON with the category's data
        geo_data = world.merge(category_data, left_on='admin', right_on='country', how='left')
        
        # Identify countries with low movie counts for the current category
        low_movie_list = low_movie_countries.get(category, [])
        geo_data['low_movie'] = geo_data['admin'].isin(low_movie_list)
        
        # Plot the merged data
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        geo_data.boundary.plot(ax=ax, linewidth=0.5, color='black')  # Plot boundaries
        
        # Plot the choropleth map
        geo_data.plot(
            column='age_score',
            ax=ax,
            legend=True,
            cmap='copper',
            vmin=global_min,  # Set global min for color scale
            vmax=global_max,  # Set global max for color scale
            missing_kwds={"color": "lightgrey", "label": "No Data"},
            legend_kwds={
                "shrink": 0.5,  # Shrink the color scale legend
                "label": "age_score",  # Add a label to the legend
                "orientation": "vertical"  # Ensure the legend is vertical
            }
        )
        
        # Overlay stripes for low-movie countries
        geo_data[geo_data['low_movie']].boundary.plot(
            ax=ax,
            linewidth=1,
            linestyle='--',  # Dashed lines for stripes
            edgecolor='red',  # Use a distinct color for stripes
            alpha=0.7
        )
        
        # Add a title
        ax.set_title(f"{category} - Age score Proportion by Country", fontsize=16)
        ax.axis('off')  # Turn off the axis for a cleaner map
        
        # Save the plot as a JPEG file
        output_filename = f"choropleth_{category.replace(' ', '_').lower()}.jpg"
        plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight')
        print(f"Saved plot for category '{category}' as '{output_filename}'.")
        plt.close(fig)  # Close the figure to free memory




