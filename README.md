# IDIA 
Extracting valuable insights from operational data is a complex challenge for organizations seeking to maximize their data assets. In addition to being a time-consuming task, the information extracted—especially in the case of unstructured sources—could not be very useful or inaccurate. To address such an issue, we propose a novel approach called IDIA, which enables technology-agnostic knowledge extraction from operational databases by utilizing Large Language Models and Vector Databases in a two-fold algorithmic process. First, IDIA indexes operational data in a vector database. Second, it leverages Large Language Models to extract relevant information and analyze the data context. To evaluate the effectiveness of our approach, we conducted a qualitative empirical study with the human resources and recruitment team of an Italian IT company. Specifically, we deployed IDIA to evaluate how it may support curriculum vitae analysis in an industrial context. The results show that IDIA effectively extracts relevant knowledge from operational data, enabling organizations to improve their decision-making processes. Feedback from the recruitment team indicates high satisfaction and potential application in other scenarios.

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
4. Set the OPENAI_API_KEY variable in config.py:

   ```
   OPENAI_API_KEY ="<your_api_key>"
   ```
4. Install poetry: `pip install poetry`
5. Create a new virtual environment with Python 3.10: `poetry env use python3.11`
6. Activate the virtual environment: `poetry shell`
7. Install app dependencies: `poetry install`
10. Run the API locally: `poetry run streamlit run ./idia/idiaApp.py`.

## About

The proof of concept is a web application which exploits the Streamlit and LangChain Python frameworks, and FAISS 10 as vector database.
It has been performed with the support of an Italian IT company. The company is an SME founded in 2002, and employs more than 50 people. It has a recruitment need and wants to evaluate the effectiveness of a new approach for extracting knowledge from resumes
using Large Language Models and Vector Databases.

