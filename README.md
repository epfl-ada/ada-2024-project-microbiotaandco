# *The evolution of diversity in film industry* :film_projector:

### *Applied Data Analysis CS-401- team MicrobiotaAndCo, EPFL* 

## Aim of the project 

The aim of this project is to analyze the evolution of diversity in the film industry, examining changes in actors’ attributes and stereotypical roles over time and across countries, as well as further investigating the potential impact of the diversity on box office revenues.

## Abstract

This project aims to analyze the evolution of diversity in the film industry by exploring variations in actors' attributes, such as gender, height, ethnicity, and age, and if and how these attributes have changed over time and territories. The project is inspired by increasing calls for better representation in media, especially from marginalized communities. 
A diversity score is developed to measure these changes, facilitating comparisons by countries and over time. In parallel, the project investigates the persistence and transformation of stereotypical roles, exploring how specific attributes may be linked to certain character archetypes. Finally, the potential impact of diversity attributes on box office revenues is examined to critically assess the notion behind the phrase 'Go woke, go broke,' which is often used to criticize progressive policies.
By clarifying these aspects, the project aims to provide insights into the cultural, demographic, and societal factors influencing representation in film. The project can provide a data-driven approach to understanding the progress of diversity in the film industry, which could inform future media production and contribute to ongoing conversations about representation and inclusivity.


## :mag_right: Research questions 

Our research question focuses on understanding how actor diversity has evolved over time and across countries, and how shifts in stereotypical roles in films are connected to these attributes across different cultures and eras.

We examine therefore whether increasing diversity in film casts and evolving character archetypes, align with the ["GO WOKE, GO BROKE?"](https://louhy12.github.io/film-reel-website/) narrative by analyzing trends over time and across countries. 

[![Spiderman_meme_GWGB](https://github.com/user-attachments/assets/1a98a0e3-d08e-4c90-8833-bc5e8f7da367)](https://louhy12.github.io/film-reel-website/)



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
├── requirements.txt            <- File for installing python dependencies
└── README.md
```
### How to use the results
Open the notebook results.ipynb and run the cells, the results will be displayed.

### The dataset

The CMU Movie Summary Corpus is a a collection of 42,306 movie plot summaries and metadata at both the movie and character level.  

This data supports work in the following paper:

David Bamman, Brendan O'Connor and Noah Smith, "Learning Latent Personas of Film Characters," in: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL 2013), Sofia, Bulgaria, August 2013.

## :computer: Methods 

The analysis was guided by several sub-questions serving as the guiding framework:
- How do we treat missing or aberrant values? How do we preprocess the dataset?
- What are the diversity attributes and how are they distributed?
- What is the relationship between and among these attributes?
- How can we define and quantify diversity? 
- How do diversity scores evolve over time and space?
- Does the diversity impact box office revenues?
- Are there common combinations of characteristics linked to specific archetypes?
- Are the results statistically significant?


0. Data preprocessing:
The aim of our preprocessing was to integrate character metadata and movie metadata while retaining only the features of interest. Additionally, a new feature, ‘Actor Country of Origin’, was created.

Unconsistent data entries were cleaned unconsistent in particular for height, age and the ethnicity. 

For the Movie Metadata, dates were recorded in varying formats, such as ‘yyyy’ and ‘yyyy.mm.dd’. Only the first four characters of each date were retained, corresponding to the year.  The mapping were extracted and retained while grouping similar values. For example, ‘Standard Cantonese’ and ‘Cantonese’ were merged into ‘Cantonese’.

The two datasets were ultimately merged into a single dataset containing the selected attributes. Non valid values were removed to ensure data quality. 

Two versions of the final dataset were saved:
    - A ‘compact’ version, where columns for countries, ethnicities, and languages contained lists of strings.
    - An ‘exploded’ version, where all list values were flattened for analysis.


1. Exploratory data analysis:

A first exploratory data analysis phase consisted in visualizing the data, analyzing the distribution of attributes, as well as their correlation. Bivariate analysis was used to explore preliminarily the presence of possible patterns.

    - Chi2 tests ??????????????????????????????????????????????

2. Data analysis: 

    - Diversity score computation:
        - We decided to calculate the gender scores with the following formula : **1-abs(female proportionality - male proportionality)**

        This score gives us the disparity between male and women. If there is no difference in the proportions, the diversity score will result being 1. If we have three quarters of men vs a quarter of woman, we will get a score of 0.5, indicating that "diversity" is defined as half-half.
        - For the height score, the age score, and the diversity score, we utilize the Simpson’s Diversity Index, which is particularly useful as it accounts for both the number of unique categories and how uniformly the categories are represented. The optimal diversity would be represented by every actor being a different ethnicity/age range/ height range from the others.
        - For the foreign actor score, diversity is measured by the proportion of actors from foreign countries. 
        - Finally, to capture overall diversity, we computed the mean of various diversity scores.

    - Diversity distribution over time and countries.
        Linear regression was employed to identify overall positive or negative trends for each diversity attribute, while polynomial regression provided a deeper time-based analysis, highlighting critical points where curvature changes sign, potentially indicating significant political or societal events. The optimal degree was selected using the highest R-squared score .... TO FINISH
 
 - Archtypes
    - Additional Dataset consisting in a new column contain stereotipycal roles. For example Batman should have at least superhero as a stereotipycal. To obtain this we queried Wikidata to obtain the occupations of a character. The query was made possible by using the Freebase Character ID. This new dataset contains 5,831 rows, and can be considered a subset of the old one. Our visualization do not suggest any biases introduced by the subselection induced by the response of Wikidata. Nevertheless additional tests may be valuable to strengthen our confidence in this new data.

    - Observational analysis:
    The analysis was done by plotting the distributions of gender, height, age, country of origin and ethnicity for each role that had at least 75 characters in the dataset.
    A Chi2 test was done to test if the repartition for each feature on all the archetypes with more than 75 characters in the dataset.


3. Statistical analysis:

    To assess diversity's evolution, yearly average or median diversity scores are analyzed using linear or polynomial regression for trends, with significance tested via R² and p-values.

    !!!!!!!!!!!!!!!!ADD MAURICE'S PAARRTTT!!!!!!!!!!!

    However, confounding factors like release dates, languages, and production countries may bias results. We propose balancing datasets using propensity score matching, reanalyzing trends, and conducting sensitivity analysis to ensure robust conclusions about diversity's economic impact.

## :hourglass: Project timeline

| Phase   | Description                                      | Date          |
|---------|--------------------------------------------------|---------------|
| Phase 1 | First dataset contact with the dataset and research question definition. | October 2024  |
| Phase 2 | Data Analysis: Analysis of diversity evolution and development of diversity scores. Analysis of the research question on the temporal and spatial level. Define dataset for the stereotypical roles. | November 2024 |
| Phase 3 | Website and data story development. Statistical analysis to test the concept of "Go woke, Go broke" and assess the impact of the diversity on box office revenues. Study of stereotypical roles evolution.| December 2024  |

## Organization within the team

Lou Houngbedji: diversity data preprocessing, website implementation ...

Matej Soumillion: stereotypical data preprocessing, diversity of archetypes analysis...

Antea Ceko: diversity data preprocessing, diversity time evolution analysis ...

Maurice Gauché: diversity data preprocessing, causal analysis diversity-box office revenues ...

Yohann Calixte Dezauzier: stereotypical data preprocessing , formulation of the diversity score, diversity contry distribution analysis...

In conclusion, we successfully established a solid foundation for collaboration in analyzing our dataset, thereby enabling us to address our research questions and drawing interesting conclusions.