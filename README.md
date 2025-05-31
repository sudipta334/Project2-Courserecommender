# Personalized Course Recommendation Engine

This project implements a Personalized Course Recommendation Engine using embeddings, semantic search, and vector similarity to suggest the most relevant courses based on user input.



## Problem Statement

Given a user query (completed courses + interest description), recommend the top-5 most relevant courses from a course catalog using embedding models and a vector database for semantic matching.



## Solution Approach

- Embeddings Generation: We use sentence-transformers (model: all-MiniLM-L6-v2) to convert course titles and descriptions into embeddings.
- Vector Search: We use FAISS as a vector database to efficiently search for top similar courses.
- Recommendation Logic: 
  - Filter out courses already completed.
  - Generate embedding for user query.
  - Compute cosine similarity to rank remaining courses.



## Dataset

File: assignment2dataset.csv

Fields:
- course_id
- title
- description



## Project Structure



Project2\_CourseRecommender/
│
├── assignment2dataset.csv         # Course catalog data
├── recommend.py                   # Main Python implementation
├── Course\_Recommendation\_Engine.ipynb  # Jupyter Notebook (optional for demo & submission)
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git exclusions (venv, pycache, etc.)
└── README.md                      # This documentation





## Setup Instructions

###  Clone the Repository

bash
git clone <your-repo-url>
cd Project2_CourseRecommender


###  Create Virtual Environment

bash
python3 -m venv venv
source venv/bin/activate


###  Install Dependencies

bash
pip install -r requirements.txt


###  Run Python Script

bash
python recommend.py


###  Run Jupyter Notebook (Optional)

bash
jupyter notebook


Open Course_Recommendation_Engine.ipynb and run all cells.



## Sample Input Queries

 "I’ve completed the ‘Foundations of Machine Learning’ course and enjoy data visualization."
 "I know Azure basics and want to manage containers and build CI/CD pipelines."
 "My background is in ML fundamentals; I’d like to specialize in neural networks and production workflows."
 "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?"
 "I’m interested in blockchain and smart contracts but have no prior experience."



## Dependencies

 pandas
 sentence-transformers
 faiss-cpu
 jupyter (optional)



## Deliverables

 Python Code
 Jupyter Notebook
 Test Outputs
 README.md
 requirements.txt



## Author

Assignment 2: Personalized Course Recommendation Engine
Delivered by: Sudipta M




