# Disaster Response Pipeline

This is a GITHUB Repository **just for educational purposes.** All data was provided by [Udacity](https://www.udacity.com/dashboard) and Figure Eight.

## Table of Contents

1. [Installation](#installation)
2. [Project Summary](#summary)
3. [Files](#files)
4. [Execution](#execution)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the ones listed in requirements.txt. The code should run with no issues using Python versions 3.12 or higher.

## Project Summary<a name="summary"></a>

The project is situated in the context of the deployment of aid organizations during major disasters (earthquakes, hurricanes, tsunamis, etc.) and aims to support targeted mission planning despite the overwhelming influx of messages to aid organizations.
The project aims to improve the accuracy of identifying critical messages compared to traditional keyword searches, which often miss relevant communications. The ultimate goal is to enhance disaster response efforts by developing new machine learning models that can better filter and prioritize important information during emergencies. 

This project involves using data engineering and machine learning skills to process and analyze pre-labeled tweets and text messages from real-life disasters. The data, provided by Figure Eight, needs to be prepared using an ETL (Extract, Transform, Load) pipeline. Afterward, a machine learning pipeline will be used to build a supervised learning model.

## Files <a name="files"></a>

### Data Files
The data is made up of two files:

* messages.csv - CSV file containing messages (if not in originally in English translated to English) exchanged related to major disasters.
* catagories.csv - labels in 38 categories related to messages.csv

### Program Files
* process_data.py - Extraction, transformation, and load of CSV above files into SQLITE DB (*.db) including data cleansing. 
* train_classifier.py - Definition and training of a supervised machine learning model based on labeled message data in SQLITE DB. Generates PICKLE (*.pkl) file containing the training model for use in web app.
* run.py - Web application with user interface to classify new messages based on user entry and model stored in PICKLE file. 

## Execution <a name="execution"></a>

Start all commands from in DisasterResponse main directory (cd DisasterResponse)

**(1) ETL**

Start of execution using command line with for mandatory arguments:

`python process_data.py [messages_filepath] [categories_filepath] [database_filepath]`

`python "./data/process_data.py" "./data/messages.csv" "./data/categories.csv" "sqlite:///./data/drp.db"`

Provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument. 

Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

**(2) Model definition and training**

Start of execution using command line with two mandatory arguments:

`python train_classifier.py [database_filepath] [model_filepath]`

`python "./models/train_classifier.py" "sqlite:///./data/drp.db" "./models/drp_classifier.pkl"`

Provide the filepath of the disaster messages database as the first argument and the filepath of the pickle  file to save the model to as the second argument. 

Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl

**(3) Web Front End**
a. Run web app: `python "./app/run.py"`
b. Open URL http://127.0.0.1:3000
