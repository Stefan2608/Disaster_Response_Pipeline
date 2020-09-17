# Disaster_Response_Pipeline

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Insturctions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
 To run the codes the following libraries need to be installed:
 
- Pandas
- Numpy
- Pickle
- Sci-kit Learn
- SQL Alchemy
- Flask
- NLTK


## Project Motivation<a name="motivation"></a>

As part of my Udacity Nanaodegree a web app was created to classify tweets during a disaster. 

## File Descriptions <a name="files"></a>

 
   * `data:`-process_data.py`ETL Pipeline cleaning and formating the data to sql database 
            -disaster_catergories.csv`provided data from Figure-8 
            -disaster_messages.csv `provided data from Figure-8 
   
   * `models -train_classifier.py` ML Pipline training a model and providing a classfier 
   
   * `app -run.py` The file used to launch a Flask web app by unsing the provided classifier.



## Instructions<a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure-8 and Udacity for the data. 
