# Streamlit Logistic Regression Project

This project implements a logistic regression model using Streamlit for interactive data visualization and model evaluation. The application allows users to load datasets, preprocess the data, train a logistic regression model, and visualize the results.

## Project Structure

```
streamlit-logistic-regression
├── app
│   ├── streamlit_app.py          # Main entry point for the Streamlit application
│   └── final_project_logistic_regression.py  # Original logistic regression code
├── src
│   ├── data_processing.py        # Data loading and preprocessing functions
│   ├── model.py                  # Model training and evaluation logic
│   └── visualization.py           # Visualization functions
├── .streamlit
│   └── config.toml               # Streamlit configuration settings
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files and directories to ignore by Git
└── README.md                      # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd streamlit-logistic-regression
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application**:
   ```
   streamlit run app/streamlit_app.py
   ```

2. **Interact with the application**:
   - Load your dataset.
   - Preprocess the data as needed.
   - Train the logistic regression model.
   - Visualize the results and performance metrics.

## Overview

This project is designed to provide a comprehensive workflow for logistic regression analysis, from data loading and preprocessing to model training and visualization. The modular structure allows for easy maintenance and scalability.