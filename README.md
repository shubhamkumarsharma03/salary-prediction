# 💼 Employee Salary Analytics Dashboard

A comprehensive machine learning web application for analyzing and predicting employee salaries in India with advanced data visualizations and model comparisons. Now available in both **monolithic** and **modular** architectures!

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

#### 🚀 Modular Version (Recommended)
```bash
streamlit run app_modular.py
```

#### 📋 Original Monolithic Version
```bash
streamlit run app_fixed.py
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

### 🏗️ Modular Architecture (New)
```
salary-prediction/
├── 🚀 app_modular.py              # Main entry point (modular version)
├── 📋 app_fixed.py                # Original monolithic version
├── 📁 config/
│   ├── __init__.py
│   ├── settings.py                # App configuration & constants
│   └── styles.py                  # CSS styles & themes
├── 📁 models/
│   ├── __init__.py
│   ├── custom_rf.py              # Enhanced Random Forest class
│   └── model_trainer.py          # Model training & evaluation
├── 📁 data/
│   ├── __init__.py
│   └── loader.py                 # Data loading & caching
├── 📁 components/
│   ├── __init__.py
│   ├── sidebar.py                # Navigation component
│   └── charts.py                 # Reusable chart functions
├── 📁 pages/
│   ├── __init__.py
│   ├── data_overview.py          # Data analysis page
│   ├── prediction.py             # Salary prediction page
│   ├── model_comparison.py       # Model comparison page
│   ├── feature_analysis.py       # Feature analysis page
│   └── model_tuning.py           # Model tuning page
├── 📁 utils/
│   └── __init__.py               # Future utilities
├── employee_salaries_india.csv   # Dataset
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── MODULAR_README.md             # Detailed modular structure docs
└── BUGFIXES.md                   # Bug fixes documentation
```

### 🎯 Architecture Benefits
- **🔧 Maintainable**: Easy to find and modify components
- **♻️ Reusable**: Components work across multiple pages
- **🧪 Testable**: Each module can be tested independently
- **👥 Collaborative**: Multiple developers can work simultaneously
- **📈 Scalable**: Simple to add new features and pages

## 🎮 Usage Guide

### 🚀 Quick Start
```bash
# Clone or download the project
cd salary-prediction

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the modular version (recommended)
streamlit run app_modular.py

# Or run the original version
streamlit run app_fixed.py
```

### 📋 Navigation Steps
1. **Start the App**: Choose either `app_modular.py` or `app_fixed.py`
2. **Navigate**: Use the sidebar to switch between analysis pages
3. **Explore Data**: Begin with "📊 Data Overview" to understand the dataset
4. **Make Predictions**: Use "🔮 Salary Prediction" with different models
5. **Compare Models**: Analyze performance in "📈 Model Comparison"
6. **Analyze Features**: Understand important factors in "🧮 Feature Analysis"
7. **Tune Models**: Experiment with parameters in "⚙️ Model Tuning"

### 🔄 Version Comparison
Both versions offer **identical functionality**:
- ✅ Same features and capabilities
- ✅ Same user interface and experience
- ✅ Same performance and caching
- ✅ Same machine learning models

**Choose based on your needs:**
- **Modular (`app_modular.py`)**: Better for development, maintenance, and collaboration
- **Monolithic (`app_fixed.py`)**: Single file, easier for deployment or sharing

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

## 🎯 Development Guidelines

### 🏗️ Modular Development (Recommended)
When working with the modular version (`app_modular.py`):

```bash
# File organization rules
pages/          → Individual page logic
components/     → Reusable UI components  
models/         → ML models & training
data/           → Data loading & processing
config/         → Settings & styles
utils/          → Helper functions
```

### � Adding New Features
1. **New Page**: Add to `pages/` directory
2. **Reusable Component**: Add to `components/`
3. **New Model**: Add to `models/`
4. **Configuration**: Update `config/settings.py`
5. **Styling**: Update `config/styles.py`

## �🚀 Future Enhancements

### 🤖 Machine Learning
1. **Advanced Models**: XGBoost, LightGBM, Neural Networks
2. **Feature Engineering**: Automated feature creation and selection
3. **Model Ensembling**: Voting and stacking classifiers
4. **Hyperparameter Optimization**: Automated tuning with Optuna

### 🎨 Frontend & UX
1. **Advanced Visualizations**: 3D plots, animated charts, interactive maps
2. **Mobile Responsive**: Better mobile experience
3. **Dark/Light Themes**: User preference themes
4. **Dashboard Customization**: User-configurable layouts

### 🔧 Technical
1. **API Development**: REST APIs for model predictions
2. **Database Integration**: PostgreSQL/MongoDB for data storage
3. **Real-time Data**: Live salary market data integration
4. **Testing Suite**: Unit tests, integration tests
5. **CI/CD Pipeline**: Automated deployment

### 📊 Analytics
1. **A/B Testing**: Feature experimentation
2. **User Analytics**: Usage pattern analysis
3. **Model Monitoring**: Performance tracking over time
4. **Data Drift Detection**: Model retraining automation

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🎯 Priority Areas
1. **Testing**: Add unit tests for modules
2. **Documentation**: Improve code documentation
3. **New Models**: Implement advanced ML algorithms
4. **UI/UX**: Enhance user interface and experience
5. **Performance**: Optimize loading and processing

### 📋 Contribution Process
1. Fork the repository
2. Create a feature branch
3. Follow the modular structure guidelines
4. Add tests for new features
5. Submit a pull request

### 🏷️ Development Setup
```bash
# Clone and setup
git clone <repository-url>
cd salary-prediction

# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
python -m pytest tests/

# Run the application
streamlit run app_modular.py
```

## 📝 License

This project is open-source and available under the MIT License.

## 📚 Additional Documentation

- **[MODULAR_README.md](MODULAR_README.md)**: Detailed modular architecture documentation
- **[BUGFIXES.md](BUGFIXES.md)**: Bug fixes and improvements log

## 🏆 Project Highlights

### ✨ **Two Architecture Approaches**
- **Monolithic** (`app_fixed.py`): Single-file application (520 lines)
- **Modular** (`app_modular.py`): Clean architecture with 6 modules

### 🎯 **Best Practices Implemented**
- **Separation of Concerns**: Each module has a single responsibility
- **Code Reusability**: Components used across multiple pages
- **Maintainability**: Easy to debug and enhance
- **Scalability**: Simple to add new features
- **Performance**: Optimized caching and processing

### 🚀 **Modern Technologies**
- **Streamlit**: Interactive web applications
- **Plotly**: Advanced data visualizations  
- **Scikit-learn**: Machine learning models
- **Pandas/NumPy**: Data processing and analysis

---

## 🎉 **Choose Your Version**

| Feature | Monolithic (`app_fixed.py`) | Modular (`app_modular.py`) |
|---------|---------------------------|--------------------------|
| **Functionality** | ✅ Complete | ✅ Complete |
| **Performance** | ✅ Optimized | ✅ Optimized |
| **Single File** | ✅ Yes | ❌ Multiple files |
| **Maintainable** | ❌ Limited | ✅ Excellent |
| **Scalable** | ❌ Difficult | ✅ Easy |
| **Team Development** | ❌ Challenging | ✅ Ideal |
| **Testing** | ❌ Complex | ✅ Simple |

**Recommendation**: Use `app_modular.py` for development and `app_fixed.py` for simple deployments.

---

**Developed with ❤️ using Streamlit, Scikit-learn, and Plotly**  
**Architecture inspired by modern web development practices** 🏗️