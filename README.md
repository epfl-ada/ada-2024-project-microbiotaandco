# *The evolution of diversity in film industry* :film_projector:

### *Applied Data Analysis CS-401- team MicrobiotaAndCo, EPFL* 

## Aim of the project 

The aim of this project is to use applied data analysis tools to analyze the evolution of diversity in film industry and explore changes in stereotypical roles over time and across different countries. 

## Abstract

This project aims to analyze the evolution of diversity in the film industry by exploring variations in actors' attributes, such as gender, height, ethnicity, and age, and if and how these attributes have changed over time.  The project is inspired by increasing calls for better representation in media, especially from marginalized communities. A diversity score will be developed to measure these changes, facilitating comparisons by country and over time. 
In parallel, the project will investigate the persistence and transformation of stereotypical roles, exploring how specific attributes may be linked to certain character archetypes.. By clarifying these aspects, the project aims to provide insights into the cultural, demographic, and societal factors influencing representation in film. 
This project is important as it can provide a data-driven approach to understanding the progress of diversity in the film industry, which could inform future media production and contribute to ongoing conversations about representation and inclusivity.


## :mag_right: Research questions 

This project is divided into two sub-analyses addressing the following general questions:

1. How has actor diversity evolved over time based on attributes such as gender, ethnicity, height, and age at the time of a movie’s release, and what trends in diversity can be observed across different countries?

2. How have stereotypical roles or archetypes in films changed over time, and how are specific attributes (e.g., gender, ethnicity) linked to character roles? Furthermore, have these patterns shifted over time and across different cultures?

To analyze these questions, we formulated several sub-questions that serve as the guiding framework for this project:

- What are the diversity attributes to be considered? 
- How are these attributed distributed? 
- How are these attributes represented in the dataset? Are there any missing or aberrant values?
- How do we treat these values? How do we preprocess the dataset?
- What is the relationship between and among these attributes? 
- How can we define and quantify diversity?
- Can we use a score? How can we implement it? What is the mathematical definition behind?
- How do diversity scores evolve over time? 
- How is diversity distributed across countries? 
- How does the diversity impact box office revenues?
- Are there attributes  that have a higher impact? 
- Are there common combinations of characteristics linked to specific archetypes?
- Are the results statistically significant? 

## :handshake: Contributors

Lou Houngbedji, Matej Soumillion, Antea Ceko, Maurice Gauché, Yohann Calixte Dezauzier.


## :open_book: Setup

- This project is available on [GitHub Pages](https://github.com/) with the following link: https://github.com/epfl-ada/ada-2024-project-microbiotaandco/tree/main

  To clone the repository from GitHub:
  ```
   cd ../path/to/the/desired/cloning/location
   git clone https://github.com/epfl-ada/ada-2024-project-microbiotaandco.git
   cd ../path/to/the/project 
   ```

- The [requirements](https://github.com/epfl-ada/ada-2024-project-microbiotaandco/blob/main/requirements.txt) file can be used to install the required libraries ```conda create -n <environment-name> --file requirements.txt```

## :open_file_folder: Project structure

### Directory structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── scripts                         <- Shell scripts
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── requirements.txt        <- File for installing python dependencies
└── README.md
```
### How to use the results
Open the notebook results.ipynb and run the cells, the results will be displayed.


## :computer: Methods 

1. Data preprocessing:
The aim of our preprocessing was to integrate character metadata and movie metadata while retaining only the features of interest: ‘Freebase Movie ID’, ‘Movie Release Date’, ‘Movie Box Office Revenue’, ‘Movie Language’, ‘Movie Country’, ‘Actor Gender’, ‘Actor Height’, ‘Actor Age’, and ‘Actor Ethnicity’. Additionally, a new feature, ‘Actor Country of Origin’, was created.

    The preprocessing steps were guided by observations made on the two datasets.

    **Character Metadata**:
    - We have cleaned unconsistent data entries in particular for height, age and the ethnicity.


    **Movie Metadata**:
For the movie metadata, we focused on cleaning the ‘Movie Release Date’, ‘Movie Language’, and ‘Movie Country’ columns.

    -    Release Date Formatting:
Dates were recorded in varying formats, such as ‘yyyy’ and ‘yyyy.mm.dd’. To standardize the data, only the first four characters of each date were retained, corresponding to the year, and converted into numeric values.

    -    Language and Country Cleaning:
The ‘Movie Language’ and ‘Movie Country’ columns contained dictionary-like structures. We extracted and retained the mappings while grouping similar values. For example, ‘Standard Cantonese’ and ‘Cantonese’ were merged into ‘Cantonese’, and ‘United Kingdom’, ‘Kingdom of Great Britain’, and ‘England’ were unified as ‘United Kingdom’. Both columns now contain lists of strings.

    **Final Dataset**:
The two datasets were ultimately merged into a single dataset containing the selected attributes. Rows with NaN values were removed to ensure data quality. Two versions of the final dataset were saved:
    - A ‘compact’ version, where columns for countries, ethnicities, and languages contained lists of strings.
    - An ‘exploded’ version, where all list values were flattened for analysis.


1. Exploratory data analysis:

    - Data form visualization
    - Attributes distribution analysis
    - Bivariate analysis 
    - Correlation analysis
    - Chi2 tests

2. Data analysis: 

    - Diversity score computation:
        - We decided to calculate the gender scores with the following formula : 
**1-abs(female proportionality - male proportionality)**
This score gives us the disparity between male and women. It takes the absolute values in difference. That way, if we have no difference in the proportions, this will mean that the difference will be 0 and we will get a score of 1, If we have three quarters of men vs a quarter of woman, we will get a score of 0.5. Meanwhile, if we have one hundred percent, there will be a score of 0. This means that ‘’ diversity’’ is defined as half-half.
        - For the height score, the age score, and the diversity score, we decide to utilize the Simpson’s diversity Index, since we are dealing with diversity in categorical feature. Simpson’s formula is particularly useful as it accounts for both richness (the number of unique categories) and evenness (how uniformly the categories are represented). Here, the optimal diversity would be represented by every actor being a different ethnicity/age range/ height range from the others.
        -For the foreign actor score, diversity is measured by the proportion of actors from foreign countries, with movies featuring only foreign actors scoring highest. To capture overall diversity, we computed the mean of various diversity scores, providing a balanced metric for comparing datasets without overemphasizing any single attribute.

    - Diversity distribution over time and countries.
 We plotted yearly mean diversity scores as a time series, applied linear regression, and observed a positive trend, indicating increasing diversity over time. Additionally, ranking countries by their mean diversity scores revealed variations, such as South Africa scoring low in ethnicity diversity and Austria excelling in gender diversity but not ethnicity.

3. Statistical analysis:

    To assess diversity's evolution, yearly average or median diversity scores are analyzed using linear or polynomial regression for trends, with significance tested via R² and p-values.

    Exploring the link between diversity and box office revenue, we found that diverse movies often outperform non-diverse ones in revenue metrics. Pearson's correlation revealed a positive association (p-value < 0.05).

    However, confounding factors like release dates, languages, and production countries may bias results. We propose balancing datasets using propensity score matching, reanalyzing trends, and conducting sensitivity analysis to ensure robust conclusions about diversity's economic impact.

5. Archtypes
- Additional Dataset: For the second question we wanted to obtain a new column that would contain stereotipycal roles. For example Batman should have at least superhero as a stereotipycal. To obtain this we queried Wikidata to obtain the occupations of a character. The query was made possible by using the Freebase Character ID. This new dataset contains 5,831 rows, and can be considered a subset of the old one. Our visualization do not suggest any biases introduced by the subselection induced by the response of Wikidata. Nevertheless additional tests may be valuable to strengthen our confidence in this new data.

- Observational analysis:
    The analysis was done by plotting the distributions of gender, height, age, country of origin and ethnicity for each role that had at least 75 characters in the dataset.
    A Chi2 test was done to test if the repartition for each feature on all the archetypes with more than 75 characters in the dataset.


## :hourglass: Proposed timeline
1. Finish the analysis (2 weeks)
    - Performing a temporality-based analysis
    - Performing a country-based analysis
    - Performing a country and temporal-based analysis
    - observational study: go woke, get broke (how diversity impacts the revenue)
    - Perform ML methods (cluster) to refine obsevational analyses
    - If we have time: Try to predict future trends on diversity
    - If we have time: Analyse how global world events (WW2, Cold war, financial crisis...) impact diversity trends in films. Ex: Russian vilains, american heroes etc...

3. Website part (2 sem)
    - Create the data story with what we want to show
    - Imagine how the interaction/animation would work
    - Code the website

## Organization within the team

The organization of work within the group was efficient and balanced, allowing each member to contribute to the project while also valuing constructive discussions and the exchange of opinions. 

Then, we divided into two groups after separating the project into two large and distinct, but intertwined, research questions. The two groups worked in parallel during the first weeks, while the final period was dedicated to comparing the analyses and results between the two groups, which allowed for a constructive exchange of opinions.
In conclusion, during this first phase, we were able to establish a solid foundation for analyzing our dataset and thus answering our research questions, while the second and final phase of the project will focus on defining the statistical significance of our results and exploring additional findings.

For the next milestone, we will organize ourselves in a similar way. This means we will analyse by teams the related questions. The organisation of the coding of the website will be split when we decided what we want to implement.

## Questions for TAs

If with further analysis that the subdataset is not very biased, is reasonable to make further analysis on it? especially "movie release date" analysis, because it is the variable being the most biased?


------------------------------------------------------------------------

This README describes data in the CMU Movie Summary Corpus, a collection of 42,306 movie plot summaries and metadata at both the movie level (including box office revenues, genre and date of release) and character level (including gender and estimated age).  This data supports work in the following paper:

David Bamman, Brendan O'Connor and Noah Smith, "Learning Latent Personas of Film Characters," in: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL 2013), Sofia, Bulgaria, August 2013.

All data is released under a Creative Commons Attribution-ShareAlike License. For questions or comments, please contact David Bamman (dbamman@cs.cmu.edu).

###
#
# DATA
#
###

1. plot_summaries.txt.gz [29 M] 

Plot summaries of 42,306 movies extracted from the November 2, 2012 dump of English-language Wikipedia.  Each line contains the Wikipedia movie ID (which indexes into movie.metadata.tsv) followed by the summary.


2. corenlp_plot_summaries.tar.gz [628 M, separate download]

The plot summaries from above, run through the Stanford CoreNLP pipeline (tagging, parsing, NER and coref). Each filename begins with the Wikipedia movie ID (which indexes into movie.metadata.tsv).


###
#
# METADATA
#
###

3. movie.metadata.tsv.gz [3.4 M]


Metadata for 81,741 movies, extracted from the Noverber 4, 2012 dump of Freebase.  Tab-separated; columns:

1. Wikipedia movie ID
2. Freebase movie ID
3. Movie name
4. Movie release date
5. Movie box office revenue
6. Movie runtime
7. Movie languages (Freebase ID:name tuples)
8. Movie countries (Freebase ID:name tuples)
9. Movie genres (Freebase ID:name tuples)



4. character.metadata.tsv.gz [14 M]

Metadata for 450,669 characters aligned to the movies above, extracted from the Noverber 4, 2012 dump of Freebase.  Tab-separated; columns:

1. Wikipedia movie ID
2. Freebase movie ID
3. Movie release date
4. Character name
5. Actor date of birth
6. Actor gender
7. Actor height (in meters)
8. Actor ethnicity (Freebase ID)
9. Actor name
10. Actor age at movie release
11. Freebase character/actor map ID
12. Freebase character ID
13. Freebase actor ID


##
#
# TEST DATA
#
##

tvtropes.clusters.txt

72 character types drawn from tvtropes.com, along with 501 instances of those types.  The ID field indexes into the Freebase character/actor map ID in character.metadata.tsv.

name.clusters.txt


970 unique character names used in at least two different movies, along with 2,666 instances of those types.  The ID field indexes into the Freebase character/actor map ID in character.metadata.tsv.
