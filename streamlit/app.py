import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    accuracy_score,
)

@st.cache_data
def load_data(classes_path, edgelist_path, features_path):
    """Load the Elliptic dataset"""
    df_classes = pd.read_csv(classes_path)
    df_edgelist = pd.read_csv(edgelist_path)
    df_features = pd.read_csv(features_path, header=None)
    return df_classes, df_edgelist, df_features

def main():
    st.title("Logistic Regression for Elliptic Transactions")
    st.markdown("Analysis of Bitcoin transactions using Logistic Regression")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    #threshold = st.sidebar.slider("Correlation Threshold", 0.5, 0.95, 0.7, 0.05)
    #pca_variance = st.sidebar.slider("PCA Variance", 0.80, 0.99, 0.95, 0.01)
    pred_threshold = st.sidebar.slider("Prediction Threshold", 0.5, 0.95, 0.8, 0.05)
    
    # File paths - Load data automatically on startup
    # Construct absolute path to data folder (one level up from streamlit folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, '..', 'data')
    
    classes_path = os.path.join(data_folder, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(data_folder, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(data_folder, "elliptic_txs_features.csv")
    
    # Load data automatically (cached after first load)
    if 'df_merged' not in st.session_state:
        try:
            # Check if files exist
            st.info(f"Looking for data files in: {data_folder}")
            if not os.path.exists(classes_path):
                st.error(f"File not found: {classes_path}")
            if not os.path.exists(features_path):
                st.error(f"File not found: {features_path}")
            if not os.path.exists(edgelist_path):
                st.error(f"File not found: {edgelist_path}")
                
            with st.spinner("Loading data..."):
                df_classes, df_edgelist, df_features = load_data(
                    classes_path, edgelist_path, features_path
                )
                
                df_features.columns = [0] + list(range(1, len(df_features.columns)))
                df_merged = df_features.merge(df_classes, left_on=0, right_on='txId', how='left')
                
                st.session_state['df_merged'] = df_merged
                st.session_state['df_features'] = df_features
                st.session_state['df_classes'] = df_classes
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Display data summary
    st.header("Data Overview")
    df_merged = st.session_state['df_merged']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", len(df_merged))
    with col2:
        st.metric("Illicit (class 1)", (df_merged['class'] == '1').sum())
    with col3:
        st.metric("Licit (class 2)", (df_merged['class'] == '2').sum())
    with col4:
        st.metric("Unknown", (df_merged['class'] == 'unknown').sum())
    with col5:
        st.metric("Suspicious", (df_merged['class'] == 'suspicious').sum())
    
    if 'df_merged' in st.session_state:
        st.header("Model Training")
        
        if st.button("Train Model"):
            try:
                with st.spinner("Training model..."):
                    st.write("**Step 1:** Merge features with classes...")
                    df_merged = st.session_state['df_merged']
                    df_features = st.session_state['df_features']
                    df_classes = st.session_state['df_classes']

                    df = df_classes.merge(
                        df_features,
                        how='left',
                        left_on='txId',
                        right_on=0
                    )

                    st.write("**Step 2:** Drop unknow class...")
                    df_labeled = df[df['class'].isin(['1', '2'])]
                    
                    st.write("**Step 3:** Splitting train/test by timestep...")
                    train_df = df_labeled[df_labeled[1].between(1, 36)]
                    test_df  = df_labeled[df_labeled[1].between(37, 49)]
                    X_train = train_df.drop(train_df.columns[0:4], axis=1)
                    X_test  = test_df.drop(test_df.columns[0:4], axis=1)
                    y_train = (train_df['class'] == '1').astype(int)
                    y_test  = (test_df['class'] == '1').astype(int)

                    
                    st.write("**Step 4:** Training Logistic Regression...")
                    lr = LogisticRegression(
                        class_weight='balanced',
                        max_iter=500
                    )
                    lr.fit(X_train, y_train)
                    st.session_state['model'] = lr
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    
                    st.success("Model trained successfully!")
                    
            except Exception as e:
                st.error(f"Error training model: {e}")
                st.exception(e)
    
    if 'model' in st.session_state:
        st.header("Model Evaluation")
        st.caption(f"Using prediction threshold: {pred_threshold}")
        
        lr = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Recalculate predictions with current threshold
        y_prob = lr.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= pred_threshold).astype(int)
        
        y_true = np.array(y_test)
        y_pred = np.array(y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{precision:.4f}")
        with col2:
            st.metric("Recall", f"{recall:.4f}")
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")


if __name__ == "__main__":
    main()