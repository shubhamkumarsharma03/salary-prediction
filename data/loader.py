"""
Data Loading and Caching Functions
"""
import streamlit as st
import pandas as pd
from config.settings import DATA_FILE

@st.cache_data
def load_data():
    """Load and cache the employee salary dataset"""
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file '{DATA_FILE}' not found!")
        return None