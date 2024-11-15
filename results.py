{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**0. Introduction**\\\n",
    "We aim to answer two main questions:\n",
    "\\\n",
    "    - How has actor diversity evolved over time based on attributes such as gender, ethnicity, height, and age at the time of a movie’s release, and what trends in diversity can be observed across different countries?\n",
    "\\\n",
    "    - How have stereotypical roles or archetypes in films changed over time, and how are specific attributes (e.g., gender, ethnicity) linked to character roles? Furthermore, have these patterns shifted over time and across different cultures?\\\n",
    "    \\\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**1. Preprocessing**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Preprocessing de Lou, Yoyo et Antea"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the second question, the main issue was that the dataset tvtropes_clusters.csv contained only 502 lines, which was too small to be used effectively for analyzing roles or archetypes. As a result, we needed to obtain new data that would include such roles or archetypes.\n",
    "\n",
    "To achieve this, we used the SPARQL endpoint API of Wikidata. By using Python code, we were able to send a SPARQL query that contained a Freebase ID for each character. This allowed us to obtain detailed information related to each character, including their associated occupations.\n",
    "\n",
    "We collected the new data and stored it in a dataset called extract_data.csv, which included a new column named \"Occupation Labels\". This column contained a list of occupations related to each character. For example, Batman had multiple occupations, such as businessman, superhero, and more.\n",
    "\n",
    "Finally, this new dataset was cleaned and processed using similar methods as for the first question. The data was normalized, duplicates were removed, and missing values were handled, resulting in a more complete and structured dataset for further analysis."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**2. Exploratory Data Analysis**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lou,Antea,Yoyo"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Maurice"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Results of the archetype analysis:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'src/data/final_data.csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[7], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01msrc\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mscripts\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01marchetype_analysis\u001b[39;00m\n",
      "File \u001b[0;32m~/Documents/EPFL/MA1/ADA_CS_401/ada-2024-project-microbiotaandco/src/scripts/archetype_analysis.py:9\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# Data import and Data manipulation\u001b[39;00m\n\u001b[1;32m      8\u001b[0m characters \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mread_csv(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msrc/data/characters_preprocessed_for_archetype.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m----> 9\u001b[0m archetypes \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mread_csv(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msrc/data/final_data.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     10\u001b[0m archetypes \u001b[38;5;241m=\u001b[39m archetypes\u001b[38;5;241m.\u001b[39mrename(columns \u001b[38;5;241m=\u001b[39m {\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mCharacter_name\u001b[39m\u001b[38;5;124m'\u001b[39m : \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mCharacter Name\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mFreebase_movie_ID\u001b[39m\u001b[38;5;124m'\u001b[39m : \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mFreebase Movie ID\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mActor_name\u001b[39m\u001b[38;5;124m'\u001b[39m : \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mActor Name\u001b[39m\u001b[38;5;124m'\u001b[39m})\n\u001b[1;32m     11\u001b[0m df \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mmerge(characters, archetypes, how \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124minner\u001b[39m\u001b[38;5;124m'\u001b[39m, on \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFreebase Movie ID\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mActor Name\u001b[39m\u001b[38;5;124m\"\u001b[39m])\n",
      "File \u001b[0;32m~/Anaconda3/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1026\u001b[0m, in \u001b[0;36mread_csv\u001b[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)\u001b[0m\n\u001b[1;32m   1013\u001b[0m kwds_defaults \u001b[38;5;241m=\u001b[39m _refine_defaults_read(\n\u001b[1;32m   1014\u001b[0m     dialect,\n\u001b[1;32m   1015\u001b[0m     delimiter,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1022\u001b[0m     dtype_backend\u001b[38;5;241m=\u001b[39mdtype_backend,\n\u001b[1;32m   1023\u001b[0m )\n\u001b[1;32m   1024\u001b[0m kwds\u001b[38;5;241m.\u001b[39mupdate(kwds_defaults)\n\u001b[0;32m-> 1026\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m _read(filepath_or_buffer, kwds)\n",
      "File \u001b[0;32m~/Anaconda3/lib/python3.12/site-packages/pandas/io/parsers/readers.py:620\u001b[0m, in \u001b[0;36m_read\u001b[0;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[1;32m    617\u001b[0m _validate_names(kwds\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnames\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m))\n\u001b[1;32m    619\u001b[0m \u001b[38;5;66;03m# Create the parser.\u001b[39;00m\n\u001b[0;32m--> 620\u001b[0m parser \u001b[38;5;241m=\u001b[39m TextFileReader(filepath_or_buffer, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n\u001b[1;32m    622\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m chunksize \u001b[38;5;129;01mor\u001b[39;00m iterator:\n\u001b[1;32m    623\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m parser\n",
      "File \u001b[0;32m~/Anaconda3/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1620\u001b[0m, in \u001b[0;36mTextFileReader.__init__\u001b[0;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[1;32m   1617\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhas_index_names\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m=\u001b[39m kwds[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhas_index_names\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[1;32m   1619\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles: IOHandles \u001b[38;5;241m|\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m-> 1620\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_make_engine(f, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mengine)\n",
      "File \u001b[0;32m~/Anaconda3/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1880\u001b[0m, in \u001b[0;36mTextFileReader._make_engine\u001b[0;34m(self, f, engine)\u001b[0m\n\u001b[1;32m   1878\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m mode:\n\u001b[1;32m   1879\u001b[0m         mode \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m-> 1880\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles \u001b[38;5;241m=\u001b[39m get_handle(\n\u001b[1;32m   1881\u001b[0m     f,\n\u001b[1;32m   1882\u001b[0m     mode,\n\u001b[1;32m   1883\u001b[0m     encoding\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mencoding\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[1;32m   1884\u001b[0m     compression\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcompression\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[1;32m   1885\u001b[0m     memory_map\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmemory_map\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mFalse\u001b[39;00m),\n\u001b[1;32m   1886\u001b[0m     is_text\u001b[38;5;241m=\u001b[39mis_text,\n\u001b[1;32m   1887\u001b[0m     errors\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mencoding_errors\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstrict\u001b[39m\u001b[38;5;124m\"\u001b[39m),\n\u001b[1;32m   1888\u001b[0m     storage_options\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstorage_options\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[1;32m   1889\u001b[0m )\n\u001b[1;32m   1890\u001b[0m \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m   1891\u001b[0m f \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles\u001b[38;5;241m.\u001b[39mhandle\n",
      "File \u001b[0;32m~/Anaconda3/lib/python3.12/site-packages/pandas/io/common.py:873\u001b[0m, in \u001b[0;36mget_handle\u001b[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[1;32m    868\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(handle, \u001b[38;5;28mstr\u001b[39m):\n\u001b[1;32m    869\u001b[0m     \u001b[38;5;66;03m# Check whether the filename is to be opened in binary mode.\u001b[39;00m\n\u001b[1;32m    870\u001b[0m     \u001b[38;5;66;03m# Binary mode does not support 'encoding' and 'newline'.\u001b[39;00m\n\u001b[1;32m    871\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m ioargs\u001b[38;5;241m.\u001b[39mencoding \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m ioargs\u001b[38;5;241m.\u001b[39mmode:\n\u001b[1;32m    872\u001b[0m         \u001b[38;5;66;03m# Encoding\u001b[39;00m\n\u001b[0;32m--> 873\u001b[0m         handle \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mopen\u001b[39m(\n\u001b[1;32m    874\u001b[0m             handle,\n\u001b[1;32m    875\u001b[0m             ioargs\u001b[38;5;241m.\u001b[39mmode,\n\u001b[1;32m    876\u001b[0m             encoding\u001b[38;5;241m=\u001b[39mioargs\u001b[38;5;241m.\u001b[39mencoding,\n\u001b[1;32m    877\u001b[0m             errors\u001b[38;5;241m=\u001b[39merrors,\n\u001b[1;32m    878\u001b[0m             newline\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    879\u001b[0m         )\n\u001b[1;32m    880\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    881\u001b[0m         \u001b[38;5;66;03m# Binary mode\u001b[39;00m\n\u001b[1;32m    882\u001b[0m         handle \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mopen\u001b[39m(handle, ioargs\u001b[38;5;241m.\u001b[39mmode)\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'src/data/final_data.csv'"
     ]
    }
   ],
   "source": [
    "import src.scripts.archetype_analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Diversity Score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import diversity_score_functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3.1 creating the diversity dataset\n",
    "In this section we calculate the diversity scores for each movie, we do this by grouping by movie, and applying formulas (simpson diversity index for age/height/age, one minus absolute difference in proportion in gender, porportions of foreign actor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_result_gender = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_gender_diversity).reset_index()\n",
    "df_result_gender.columns = ['Freebase Movie ID', 'gender_score']\n",
    "df_result_gender.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_result_foreigners = calculate_foreign_actor_proportion(df_metadata_OI)\n",
    "df_result_foreigners.head(10)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_result_age = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_age_diversity).reset_index()\n",
    "df_result_age.columns = ['Freebase Movie ID', 'age_score']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_result_height = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_height_diversity).reset_index()\n",
    "df_result_height.columns = ['Freebase Movie ID', 'height_score']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_result_ethnicity = df_metadata_OI.groupby('Freebase Movie ID').apply(calculate_ethnicity_diversity).reset_index()\n",
    "df_result_ethnicity.columns = ['Freebase Movie ID', 'ethnicity_score']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section, we create a new data set where each movie posseses a score. Additionally we add for each movie the overall diversity score which is just the mean of the previous scores combined."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_merged = df_result_age \\\n",
    "    .merge(df_result_height, on='Freebase Movie ID') \\\n",
    "    .merge(df_result_ethnicity, on='Freebase Movie ID') \\\n",
    "    .merge(df_result_gender, on='Freebase Movie ID')\\\n",
    "    .merge(df_result_foreigners, on='Freebase Movie ID')\n",
    "\n",
    "df_merged['diversity_score'] = df_merged[['age_score', 'height_score', 'ethnicity_score', 'gender_score','Foreign Actor Proportion']].mean(axis=1)\n",
    "df_merged.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Diversity_movie_metadata=df_merged.merge(\n",
    "    df_merged_unique[['Freebase Movie ID', 'Movie Release Date', 'Movie Box Office Revenue', 'Movie Language', 'Movie Country']],\n",
    "    on='Freebase Movie ID',\n",
    "    how='inner') \n",
    "Diversity_movie_metadata.sample(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3.2 preliminary analysis of the dataset\n",
    "In this section, we decided to plot the evolution of the various diversity scores over time and over the different countries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import plot_diversity_country_time"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
