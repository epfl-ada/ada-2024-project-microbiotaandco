# *Title: The evolution of diversity in film industry :film_projector:*

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


### How to use the library


## :computer: Methods 

1. Data preprocessing:

2. Exploratory data analysis:

- Data form visualization
- Attributes distribution analysis
- Bivariate analysis 
- Correlation analysis
- Chi2 tests

3. Data analysis: 

- Diversity score computation:
- We decided to calculate the gender scores with the the formula :
-)1-abs(pf-ph).
This score gives us the disparity between male and women. It takes the absolute values in difference. That way, if we have no difference in the proportions, this will mean that the difference will be 0 and we will get a score of 1, If we have three quarters of men vs a quarter of woman, we will get a score of 0.5. Meanwhile, if we have one hundred percent, there will be a score of 0. This means that ‘’ diversity’’ is defined as half-half.
-For the height score, the age score, and the diversity score, we decide to utilize the Simpson’s diversity Index, since we are dealing with diversity in categorical feature. Simpson’s formula is particularly useful as it accounts for both richness (the number of unique categories) and evenness (how uniformly the categories are represented). Here, the optimal diversity would be represented by every actor being a different ethnicity/age range/ height range from the others.
-For the foreign actor score, we have a metric where more foreign actors means more diversity. As such, there is a very obvious metric we could use:
The proportion of actors whose country of origin. A movie with only foreign actors will get a score.
Finally, to compute the mean score, we decided to compute the mean of the various diversity scores.
By averaging, we can capture a holistic sense of diversity across various attributes,  it is easier to compare datasets or groups on an equal footing without overemphasizing any single category.

- Diversity distribution over time and countries.
  We took the mean diversity scores for each year, and proceeded to plot them as time series and performed linear regression on them. Calculating the mean diversity score for each year helped us identify overarching trends in diversity over time, smoothing out fluctuations caused by individual outliers or specific movies in a given year The thing we notice is that ,unsurprisingly, the coefficients are positive for most diversity score over time, so diversity seems to increase over time.

We also decided  that amongst the  30 most represented countries, we would calculate the mean diversity score of each country, and then rank them from largest to lowest. This allows us to observe that the distribution is different amongst countries. For instance, South Africa has a very low ethnicity score while Brazil has a very high one. The second observation is that the distributions are different for the categories. Austria scores high for gender diversity but not as high for ethnicity diversity.


4. Statistical analysis:

In order to study if diversity has evolved over time and its evolution, it is possible to analyze its trajectory with the average or median diversity scores per year. This can be done by plotting the diversity scores over time (years), which allows for the identification of an eventual trend. Once a trend is identified, if there is one, linear regression can be applied to model the relationship between time and a specific diversity. Linear regression can help in providing insights into whether the trend is increasing or decreasing overall (slope of the regression line).

If no trend can be established with linear regression or if it is not significant, polynomial regression can be used to capture more complex, non-linear trends. 

After establishing a possible trend, a statistical assessment is necessary to evaluate the significance of it. This can be done using the R² score to assess how well the model explains the variability in diversity scores (higher R² value means the model is a good fit). 
Additionally, statistical tests such as t-tests or ANOVA can be used to verify if the observed trend is statistically significant, excluding the possibility that the trend occurred randomly. The p-value obtained from the regression model will be used to test the null hypothesis that there is no trend in diversity over time.


In recent years, the phrase "Get woke, go broke" has emerged as a contentious expression, suggesting that efforts by media companies to prioritize diversity in casting may come at a financial cost. 
We decided to conduct an observational study on the causal relationship between the overall diversity score and the Movie Box Office Revenue.
We first decided to split the dataset in half : the ‘’diverse’’ set in the upper half  and the non diverse set with their diversity score below the median.
After this, we could perform a naive analysis, by plotting the distribution of the two set’s movie revenues. We also observe the means, quartiles and medians, and the overall results seem to point out that diverse movies tend to fare better economically. We observe that  the mean, quartiles, min and median box office revenues are all higher for the diverse set.
We use pearsons correlation. it is well-suited for examining the relationship between diversity scores and box office revenue because it quantifies the strength and direction of the linear association between these two continuous variables. By using Pearson’s correlation, we can directly assess whether higher diversity scores are consistently associated with changes in revenue, whether positive or negative. With a p-value below 0.05, we reject the null hypothesis that the two values are independent.
However, this is a naive analysis. We are failing to take into account observable factors that might affect the probability of a movie being diverse such as :
-movie release date
-Movie countries
-movie languages.
We observe through the bar plots that the diverse dataset has a larger share of movies with more than one language and more than one country. It also has more movies with very high language counts. This is problematic as these might be factors that influence both the chances of a movie being diverse and the revenue, and might therefore bias our conclusion.
The boxplot shows us the quartiles of the non diverse movies tend to be older. This might also be problematic as we need to account for the fact that modern movies might make more money  and modern movies might be more diverse.
We then propose to rebalance the datasets by calculating a propensity score, and do a match 1 to 1 of the dataset. This way we will eliminate these three potential factors from the causal links.
We can then reperform the analysis and see if the results of the analysis will be proven.
Finally we must consider the unseen factors : We propose to perform a sensitivity analysis in order to take them into account. 



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

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

:white_square_button:

## Organization within the team

The organization of work within the group was efficient and balanced, allowing each member to contribute to the project while also valuing constructive discussions and the exchange of opinions. 

More specifically, we started by choosing a project that would allow us to have a dynamic and interesting analysis. Then, we divided into two groups after separating the project into two large and distinct, but intertwined, research questions. The first phase of individual analysis allowed us to collect various methods for handling the initial data and choose the paths most aligned with the research question. Afterward, we were able to discuss which data analysis methods would help us obtain results in line with our questions and begin collecting results. 

The two groups worked in parallel during the first weeks, while the final period was dedicated to comparing the analyses and results between the two groups, which allowed for a constructive exchange of opinions.
In conclusion, during this first phase, we were able to establish a solid foundation for analyzing our dataset and thus answering our research questions, while the second and final phase of the project will focus on defining the statistical significance of our results and exploring additional findings.

For the next milestone, we will organize ourselves in a similar way. This means we will analyse by teams the related questions. The organisation of the coding of the website will be split when we decided what we want to implement.

## Questions for TAs




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
