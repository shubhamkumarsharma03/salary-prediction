# 💼 Employee Salary Analytics Dashboard

A comprehensive machine learning web application for analyzing and predicting employee salaries in India with advanced data visualizations and model comparisons.

## 🚀 Features

### 📊 Enhanced Frontend
- **Multi-page Navigation**: Organized into 5 distinct sections for better user experience
- **Interactive Visualizations**: Using Plotly for dynamic, interactive charts
- **Responsive Design**: Wide layout with custom CSS styling
- **Professional UI**: Card-based metrics, gradient backgrounds, and intuitive controls

### 🤖 Advanced Machine Learning
- **Multiple Model Comparison**: Custom Random Forest, Scikit-Learn RF, Decision Tree, and Linear Regression
- **Cross-Validation**: K-fold cross-validation for robust model evaluation
- **Feature Importance Analysis**: Understand which factors most impact salary predictions
- **Hyperparameter Tuning**: Interactive parameter adjustment with real-time results

### 📈 Deep Analysis Pages

#### 1. 📊 Data Overview
- Dataset statistics and key metrics
- Interactive salary distribution histograms
- Experience vs Salary scatter plots with education coloring
- Average salary analysis by location and job role
- Correlation heatmap for numeric features

#### 2. 🔮 Salary Prediction
- Enhanced prediction interface with input summary
- Multiple model selection for predictions
- Confidence metrics (R², RMSE) display
- Salary comparison with similar profiles in the market
- Professional prediction result cards

#### 3. 📈 Model Comparison
- Side-by-side performance comparison of all models
- Training vs Testing performance visualization
- Detailed metrics table with formatted currency values
- R² Score and RMSE comparisons with grouped bar charts

#### 4. 🧮 Feature Analysis
- Top 15 most important features visualization
- Statistical analysis by experience ranges
- Education level impact analysis
- Feature importance rankings

#### 5. ⚙️ Model Tuning
- Interactive hyperparameter adjustment
- Real-time cross-validation results
- Custom model training with user-defined parameters
- Prediction vs Actual scatter plots for model validation

## 🎯 Key Improvements Over Original

### Frontend Enhancements
1. **Multi-page Architecture**: Organized content into logical sections
2. **Professional Styling**: Custom CSS with modern design elements
3. **Interactive Charts**: Plotly visualizations instead of static plots
4. **Better UX**: Intuitive navigation and responsive layout
5. **Rich Metrics Display**: Professional metric cards and result presentations

### Machine Learning Improvements
1. **Model Ensemble**: Multiple algorithms for comparison
2. **Enhanced Custom Classes**: Added feature importance calculation
3. **Cross-Validation**: Robust model evaluation with K-fold CV
4. **Performance Metrics**: Comprehensive evaluation (R², RMSE, MAE)
5. **Hyperparameter Tuning**: Interactive parameter optimization

### Data Analysis Features
1. **Comprehensive EDA**: Statistical analysis and visualizations
2. **Feature Importance**: Understanding model decisions
3. **Market Comparison**: Salary benchmarking for similar profiles
4. **Correlation Analysis**: Feature relationship exploration
5. **Distribution Analysis**: Detailed data distribution insights

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+ required
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run enhanced_app.py
```

### Required Packages
- streamlit==1.28.0
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2
- plotly==5.15.0

## 📁 Project Structure

```
├── enhanced_app.py          # Main enhanced application
├── app.py                   # Original application
├── employee_salaries_india.csv  # Dataset
├── requirements.txt         # Package dependencies
└── README.md               # This file
```

## 🎮 Usage Guide

1. **Start the App**: Run `streamlit run enhanced_app.py`
2. **Navigate**: Use the sidebar to switch between analysis pages
3. **Explore Data**: Begin with "Data Overview" to understand the dataset
4. **Make Predictions**: Use "Salary Prediction" with different models
5. **Compare Models**: Analyze performance in "Model Comparison"
6. **Analyze Features**: Understand important factors in "Feature Analysis"
7. **Tune Models**: Experiment with parameters in "Model Tuning"

## 🔧 Technical Architecture

### Enhanced Custom Classes
- **EnhancedDecisionTreeRegressor**: Added feature importance calculation
- **EnhancedRandomForestRegressor**: Ensemble feature importance and better handling

### Pipeline Components
- **Preprocessing**: StandardScaler for numeric, OneHotEncoder for categorical
- **Model Training**: Multiple algorithms with cross-validation
- **Evaluation**: Comprehensive metrics calculation
- **Visualization**: Interactive Plotly charts throughout

### Performance Features
- **Caching**: `@st.cache_data` for expensive operations
- **Efficient Processing**: Sparse matrix handling for memory optimization
- **Real-time Updates**: Dynamic parameter tuning with immediate feedback

## 📊 Dataset Information

**Employee Salaries India Dataset**
- **Records**: 150 employees
- **Features**: 7 input features + 1 target (salary)
- **Target**: Salary in INR (Indian Rupees)
- **Features**: Experience, Education, Location, Company Size, Industry, Job Role, Remote Work

## 🎯 Model Performance

The enhanced application provides detailed model comparison with metrics:
- **R² Score**: Model explanation power
- **RMSE**: Root Mean Square Error in currency
- **MAE**: Mean Absolute Error for interpretability
- **Cross-Validation**: K-fold validation scores

## 🚀 Future Enhancements

1. **Advanced Models**: XGBoost, LightGBM integration
2. **Feature Engineering**: Automated feature creation
3. **Model Deployment**: API endpoints for production use
4. **Real-time Data**: Integration with live salary databases
5. **Advanced Visualizations**: 3D plots and animated charts

## 🤝 Contributing

Feel free to contribute by:
1. Adding new visualization features
2. Implementing additional ML algorithms
3. Improving the UI/UX design
4. Adding more comprehensive analysis features

## 📝 License

This project is open-source and available under the MIT License.

---

**Developed with ❤️ using Streamlit, Scikit-learn, and Plotly**