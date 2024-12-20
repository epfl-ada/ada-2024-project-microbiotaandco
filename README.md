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

We examine therefore whether increasing diversity in film casts and evolving character archetypes, align with the ["GO WOKE, GO BROKE?"](https://louhy12.github.io/film-reel-website/go-woke-go-broke-really/) narrative by analyzing trends over time and across countries. 

[![Spiderman_meme_GWGB](https://github.com/user-attachments/assets/1a98a0e3-d08e-4c90-8833-bc5e8f7da367)](https://louhy12.github.io/film-reel-website/go-woke-go-broke-really/)



## :handshake: Contributors

Lou Houngbedji, Matej Soumillion, Antea Ceko, Maurice Gauché, Yohann Dezauzier.


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


1. Data preprocessing:
The aim of our preprocessing was to integrate character metadata and movie metadata while retaining only the features of interest. Additionally, a new feature, ‘Actor Country of Origin’, was created.

Unconsistent data entries were cleaned unconsistent in particular for height, age and the ethnicity. 

For the Movie Metadata, dates were recorded in varying formats, such as ‘yyyy’ and ‘yyyy.mm.dd’. Only the first four characters of each date were retained, corresponding to the year.  The mapping were extracted and retained while grouping similar values. For example, ‘Standard Cantonese’ and ‘Cantonese’ were merged into ‘Cantonese’.

The two datasets were ultimately merged into a single dataset containing the selected attributes. Non valid values were removed to ensure data quality. 

Two versions of the final dataset were saved:
    - A ‘compact’ version, where columns for countries, ethnicities, and languages contained lists of strings.
    - An ‘exploded’ version, where all list values were flattened for analysis.


2. Exploratory data analysis:

The initial exploratory data analysis phase involved visualizing the data through univariate analysis, examining the distribution of attributes and their correlations.

3. Data analysis: 

 3.1. Diversity analysis

- Diversity scores computation: 
We decided to calculate the diversity scores(1 being highest and 0 being lowest) for each movie with Simpson diversity index except for the foreign actors where we used proportion of foreigner and gender score where we used 1-|pfemale-pmale|. We then averaged all scores to create the diversity score. We then averaged them by country and time period to see a country's average performance at each time and create maps that could show trends.

- The diversity distribution over time and countries:
Linear regression was employed to identify overall positive or negative trends for each diversity attribute, while polynomial regression provided a deeper time-based analysis, highlighting critical points where curvature changes sign, potentially indicating significant political or societal events. The optimal degree was selected using the highest R-squared score.


3.2. Causal analysis: diversity influence on box office revenues

We split the data into diverse and non diverse  and observed movie revenues. We made sure the datasets were balanced for causal inference using propensity score matching and we also conducted a sensitivity analysis.

Another approach to account for control variables is to use residual regression. The idea is to regress linearly the variables of interest, that is diversities scores and Box office revenue on all control variables and then use the obtained residuals to conduct a second linear regression. Residuals are the part of variables that are not explained by the control variables, so comparing them might give insight about direct relationships between the Box office and the different diversity scores.


3.3. Archetype analysis

The analysis was done by bar chart race videos. We only kept archetypes with more than 50 characters related to them. A chi-square test was done for gender and ethnicities repartitions both for the whole set of archetypes and for each individual one.


3.4. Causal analysis: diversity influence on box office revenues

We wanted to find out if the disparities in the archetypes lead to a significant change in the revenue. For this we plotted the p-values of a linear regression predicting revenue with archetypes against the p-values of the chi-square tests representing the disparities in archetypes for gender and ethnicity. We then made a linear regression on this data, giving us insights on the significance of archetypes’ disparities on the change in the revenue.


## :hourglass: Project timeline

| Phase   | Description                                      | Date          |
|---------|--------------------------------------------------|---------------|
| Phase 1 | First dataset contact with the dataset and research question definition. | October 2024  |
| Phase 2 | Data Analysis: Analysis of diversity evolution and development of diversity scores. Analysis of the research question on the temporal and spatial level. Define dataset for the stereotypical roles. | November 2024 |
| Phase 3 | Website and data story development. Statistical analysis to test the concept of "Go woke, Go broke" and assess the impact of the diversity on box office revenues. Study of stereotypical roles evolution.| December 2024  |

## Organization within the team

Lou Houngbedji: diversity data preprocessing, diversity exploratory data analysis (EDA), website implementation.


Matej Soumillion: archetypes visualization and archetypes analysis, archetypes causal analysis.


Antea Ceko: diversity EDA, diversity time evolution and historical analysis, REAME.


Maurice Gauché: archetypes data preprocessing, diversity-box office revenues causal analysis.


Yohann Dezauzier: causal inference analysis, diversity score implementation, diversity country distribution analysis.


In conclusion, we successfully established a solid foundation for collaboration in analyzing our dataset, thereby enabling us to address our research questions and drawing interesting conclusions.