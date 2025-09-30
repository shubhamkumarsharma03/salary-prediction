"""
Navigation Sidebar Component
"""
import streamlit as st
from config.settings import NAVIGATION_OPTIONS

def render_sidebar():
    """Render the navigation sidebar"""
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", NAVIGATION_OPTIONS)
    return page