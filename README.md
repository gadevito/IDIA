# IDIA 
Extracting valuable insights from operational data is a complex challenge for organizations seeking to maximize their data assets. In addition to being a time-consuming task, the information extracted—especially in the case of unstructured sources—could not be very useful or inaccurate. To address such an issue, we propose a novel approach called IDIA. 
To evaluate the effectiveness of our approach, we conducted a qualitative empirical study with the human resources and recruitment team of an Italian IT company. Specifically, we deployed IDIA to evaluate how it may support curriculum vitae analysis in an industrial context. 

## Introduction

The IDIA repository provides the proof of concept of the IDIA approach, used to assess its effectiveness through a small-scale industrial empirical study conducted with an Italian IT company. The repository is organized into several directories:

| Directory                       | Description                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [`idia`](/idia)       | Contains the core implementation of the IDIA algorithms and the case study Web Application.              |
| [`running-example`](/running-example)                 | Contains the data used for the running example reported in the paper. |


This README provides detailed information on how to set up and deploy the IDIA proof of concept.

## Table of Contents

- [Quickstart](#quickstart)
- [About](#about)

## Quickstart

Follow these steps to quickly set up and run the IDIA proof of concept:

1. Install Python 3.11, if not already installed.
2. Clone the repository: `git clone https://github.com/gadevito/IDIA.git`
3. Navigate to the cloned repository directory: `cd /path/to/IDIA`
4. Create the data directory: `mkdir data`
5. Set the OPENAI_API_KEY variable in config.py, in the /path/to/IDIA/idia directory:

   ```
   OPENAI_API_KEY ="<your_api_key>"
   ```
6. Navigate again to the cloned repository directory: `cd /path/to/IDIA`
7. Install poetry: `pip install poetry`
8. Create a new virtual environment with Python 3.11: `poetry env use python3.11`
9. Activate the virtual environment: `poetry shell`
10. Install app dependencies: `poetry install`
11. Run the API locally: `poetry run streamlit run ./idia/idiaApp.py`.

## About

The proof of concept is a web application which exploits the Streamlit and LangChain Python frameworks, and FAISS 10 as vector database.
It has been performed with the support of an Italian IT company. The company is an SME founded in 2002, and employs more than 50 people. It has a recruitment need and wants to evaluate the effectiveness of a new approach for extracting knowledge from resumes
using Large Language Models and Vector Databases.
