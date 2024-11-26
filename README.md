# Loan approval service

## Overview

This project involves building a machine learning model to predict the approval status
of loan applications based on applicants' financial, demographic,
and credit-related information. Given the significant risk and regulatory implications
associated with credit approvals, an accurate model will help streamline decision-making,
reduce financial risk, and improve the customer experience by automating the approval process.

## Problem Statement

Loan providers receive numerous applications, each with varying levels
of financial reliability. Reviewing these applications manually is time-consuming and
may lack consistency. Our goal is to automate this process by creating a model
that predicts the likelihood of an applicant being approved or denied based on historical data,
thereby increasing efficiency and accuracy in credit approval decisions.

**The next dataset will be used for to predict loan approval:**

https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data

Target value - **loan status** variable.

Loan status **Approved** will be set if calculated 'probability rate' >= 0.75
otherwise, the status will be **Rejected**.

Dataset [archive.zip](datasets%2Farchive.zip) is located in **datasets** folder.

##### 1. How To get the data

Execute **_Data extraction_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb)
to extract the data ([loan_data.csv](datasets%2Floan_data.csv) will be extracted).

##### 2. Perform Exploratory data analysis (EDA)

Execute **_Exploratory data analysis (EDA)_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb)
to perform EDA(check duplicates, missing values, normalize the columns, etc.).

##### 3. Train the model

Run command from project root folder: <p> 
`python ./scripts/train.py`

File [dv_model.pkl](datasets%2Fdv_model.pkl) will be created. It is also possible to
specify particular dataset as input and file with model as output:
`python ./scripts/train.py --input <path to dataset> --out <path to model>`

Example:
`python ./scripts/train.py --input './datasets/loan_data.csv' --out './datasets/dv_model.pkl'`

##### 4. Predict

Loan approval service is defined in [train.py](scripts%2Ftrain.py).

To run the service you need to perform next steps:
- build docker image from the next [Dockerfile](Dockerfile) - `docker build -t loan_approval_service .`
- run docker container `docker run -it -p 8024:8024 loan_approval_service:latest`


##### 5. Using Loan approval service

- Execute all actions defined in section 4.
- Use data and code defined in  **_Test service_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb)
