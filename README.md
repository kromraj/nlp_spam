# Natural Language Processing (NLP) - Spam Detection
## Introduction
This project focuses on spam detection using Natural Language Processing (NLP) techniques. The dataset comprises spam-related text data, and the goal is to prepare the text, train a Random Forest model, extract feature importance, perform feature selection based on importance, and then train a new Random Forest model with hyperparameter tuning.

## Steps
1. **Text Data Preprocessing**
- Removal of punctuation marks
- Tokenization
- Removal of stopwords
- Stemming
- Lemmatization
- TF-IDF Vectorization
- CountVectorizer()
2. **Data Splitting and Model Training**
- Split the dataset into training and testing sets.
- Train a Random Forest model and extract feature importance.
- Perform feature selection by considering features with importance greater than 0.001.
3. **Hyperparameter Tuning and Model Evaluation**
- Train a new Random Forest model with hyperparameter tuning using GridSearch.
- Evaluate metrics and confusion matrix for the trained model.
## Technologies Used
- **Python:** Primary programming language for NLP and machine learning tasks.
- **Pandas:** Library for data manipulation and analysis.
- **Scikit-Learn:** Library for machine learning models, tools, and GridSearch.
- **NLTK (Natural Language Toolkit):** Library for natural language processing tasks.
- **Matplotlib:** Library for creating static, animated, and interactive visualizations in Python.
- **Seaborn:** Data visualization library based on Matplotlib, providing additional functionality and aesthetically pleasing visualizations.
