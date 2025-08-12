# TimeSeries-Portfolio 📊⏰

A comprehensive time series analysis and portfolio optimization framework built with Python. This project provides tools for data preprocessing, exploratory data analysis, forecasting, portfolio optimization, and backtesting with a focus on financial time series data.

## 🚀 Features

### Data Management
- **Data Loading**: Support for multiple data sources (CSV, APIs, databases)
- **Data Preprocessing**: Cleaning, normalization, feature engineering
- **Data Validation**: Comprehensive data quality checks
- **Financial Data**: Built-in support for stock data via yfinance

### Time Series Analysis
- **Exploratory Data Analysis**: Statistical summaries, visualizations, pattern detection
- **Stationarity Testing**: ADF, KPSS tests with automatic differencing
- **Seasonal Decomposition**: Trend, seasonal, and residual analysis
- **Correlation Analysis**: Asset correlation matrices and heatmaps

### Forecasting Models
- **ARIMA Models**: Auto-regressive integrated moving average with automatic parameter selection
- **LSTM Models**: Long short-term memory neural networks for complex patterns (PyTorch-based)
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Model Evaluation**: Comprehensive performance metrics and diagnostics
- **Model Persistence**: Save and load trained models for future use

### Portfolio Optimization
- **Risk-Return Optimization**: Maximum Sharpe ratio, minimum variance portfolios
- **Efficient Frontier**: Generate optimal portfolio combinations
- **Asset Allocation**: Equal weight, value-weighted, and maximum diversification strategies
- **Risk Metrics**: VaR, CVaR, maximum drawdown analysis
- **Constraint Handling**: Support for various optimization constraints

### Backtesting Framework
- **Strategy Testing**: Buy & hold, momentum, mean reversion strategies
- **Performance Analysis**: Returns, risk metrics, drawdown analysis
- **Transaction Costs**: Realistic trading simulation with costs and slippage
- **Strategy Comparison**: Multi-strategy performance evaluation
- **Performance Reports**: Comprehensive backtesting summaries

## 📁 Project Structure

```
TimeSeries-Portfolio/
│
├── data/                          # Raw and processed data
│   ├── raw/                      # Original data files
│   └── processed/                # Cleaned and transformed data
│       ├── *_enhanced_data.csv   # Preprocessed financial data
│       ├── model_predictions.csv # Forecasting results
│       ├── backtesting_results.pkl # Backtesting outputs
│       └── risk_metrics_summary.csv # Risk analysis results
│
├── notebooks/                     # Jupyter Notebooks for each task
│   ├── 01_preprocessing_eda.ipynb      # Data preprocessing and EDA
│   ├── 02_forecasting_models.ipynb     # ARIMA and LSTM models
│   ├── 03_future_forecast_analysis.ipynb # Future forecasting analysis
│   ├── 04_portfolio_optimization.ipynb  # Portfolio optimization
│   └── 05_backtesting.ipynb            # Strategy backtesting
│
├── src/                           # Python modules for reusable code
│   ├── data_loader.py            # Data loading and management
│   ├── preprocessing.py          # Data cleaning and transformation
│   ├── eda.py                    # Exploratory data analysis
│   ├── arima_model.py            # ARIMA model implementation
│   ├── lstm_model.py             # LSTM model implementation (PyTorch)
│   ├── forecasting.py            # Unified forecasting interface
│   ├── portfolio_optimization.py # Portfolio optimization tools
│   ├── backtesting.py            # Backtesting framework
│   ├── arima_model.pkl          # Saved ARIMA model
│   ├── lstm_model.h5            # Saved LSTM model
│   └── lstm_scaler.pkl          # LSTM data scaler
│
├── reports/                       # Output reports and visualizations
│   ├── figures/                  # Generated plots and charts
│   └── summary_report.md         # Project summary report
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # Project license
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (Note: TensorFlow/Keras not compatible with Python 3.13)
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/TimeSeries-Portfolio.git
   cd TimeSeries-Portfolio
   ```

2. **Create a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, matplotlib, seaborn, torch; print('Installation successful!')"
   ```

### Key Dependencies

- **Core**: pandas, numpy, scipy, scikit-learn
- **Statistics**: statsmodels
- **Deep Learning**: PyTorch (TensorFlow alternative for Python 3.13+)
- **Visualization**: matplotlib, seaborn, plotly
- **Financial**: yfinance, pandas-datareader
- **Optimization**: cvxpy, pulp
- **Jupyter**: jupyter, ipykernel

## 📊 Quick Start

### 1. Data Preprocessing and EDA
```python
from src.data_loader import DataLoader
from src.preprocessing import preprocess_pipeline
from src.eda import run_complete_eda

# Load financial data
loader = DataLoader()
data = loader.load_csv('data/processed/SPY_enhanced_data.csv')

# Preprocess data
processed_data, summary = preprocess_pipeline(
    data,
    handle_missing=True,
    remove_outliers=True,
    normalize=True,
    create_lags=True
)

# Run comprehensive EDA
eda_results = run_complete_eda(processed_data)
```

### 2. Time Series Forecasting
```python
from src.forecasting import run_complete_forecasting

# Run complete forecasting pipeline
forecasting_engine = run_complete_forecasting(
    data['close'], 
    forecast_steps=30,
    models=['arima', 'lstm']
)

# Access results
forecasts = forecasting_engine.forecasts
ensemble_forecast = forecasting_engine.ensemble_forecast
model_performance = forecasting_engine.model_performance
```

### 3. Portfolio Optimization
```python
from src.portfolio_optimization import run_portfolio_optimization

# Create returns data from multiple assets
returns_data = data[['SPY', 'TSLA', 'BND']].pct_change().dropna()

# Run portfolio optimization
optimizer = run_portfolio_optimization(
    returns_data, 
    risk_free_rate=0.02,
    optimization_method='sharpe_ratio'
)

# Access results
optimal_weights = optimizer.optimal_weights
efficient_frontier = optimizer.efficient_frontier
risk_metrics = optimizer.risk_metrics
```

### 4. Strategy Backtesting
```python
from src.backtesting import run_complete_backtesting

# Run comprehensive backtesting
backtester = run_complete_backtesting(
    market_data, 
    initial_capital=100000,
    strategies=['buy_hold', 'momentum', 'mean_reversion']
)

# Compare strategies
comparison = backtester.compare_strategies()
best_strategy = backtester.get_best_strategy('sharpe_ratio')
performance_report = backtester.generate_report()
```

## 📚 Jupyter Notebooks

The project includes comprehensive Jupyter notebooks that walk through each analysis step:

1. **01_preprocessing_eda.ipynb** - Data cleaning, exploration, and visualization
2. **02_forecasting_models.ipynb** - ARIMA and LSTM model implementation
3. **03_future_forecast_analysis.ipynb** - Long-term forecasting and analysis
4. **04_portfolio_optimization.ipynb** - Risk-return optimization strategies
5. **05_backtesting.ipynb** - Strategy evaluation and performance analysis

Each notebook includes:
- Detailed explanations of concepts
- Step-by-step implementation
- Visualization and analysis
- Results interpretation
- Best practices and tips

## 🔧 Configuration

### Data Sources
- Update data paths in `src/data_loader.py`
- Configure API keys for external data sources (yfinance, etc.)
- Set data format preferences and column mappings

### Model Parameters
- Adjust ARIMA parameter ranges in `src/arima_model.py`
- Modify LSTM architecture in `src/lstm_model.py`
- Configure optimization constraints in `src/portfolio_optimization.py`
- Set forecasting horizons and validation periods

### Backtesting Settings
- Set transaction costs and slippage in `src/backtesting.py`
- Configure strategy parameters and thresholds
- Adjust performance metrics and reporting options

## 📈 Output and Reports

The framework generates comprehensive outputs:

- **Visualizations**: Time series plots, correlation matrices, performance charts, efficient frontiers
- **Reports**: Markdown reports with analysis summaries and recommendations
- **Data Files**: Processed datasets, model results, optimization outputs
- **Models**: Saved trained models for future use (ARIMA, LSTM)
- **Performance Metrics**: Risk-adjusted returns, drawdown analysis, strategy comparisons

## 🚨 Important Notes

### Python Version Compatibility
- **Python 3.13+**: TensorFlow/Keras are not compatible
- **Recommended**: Use PyTorch for deep learning (included in requirements.txt)
- **Alternative**: XGBoost and LightGBM for gradient boosting

### Model Persistence
- ARIMA models are saved as `.pkl` files
- LSTM models are saved as `.h5` files
- Data scalers are preserved for consistent preprocessing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with popular Python data science libraries
- Inspired by financial time series analysis best practices
- Designed for educational and research purposes
- Special thanks to the open-source community

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the notebooks
- Review the source code examples
- Consult the comprehensive error handling in each module

## 🔮 Future Enhancements

- Additional forecasting models (Prophet, NeuralProphet)
- Real-time data streaming capabilities
- Web dashboard interface (Streamlit/Dash)
- Advanced risk management tools
- Multi-asset class support
- Alternative data integration
- Cloud deployment support
- API endpoints for model serving

## 📊 Sample Results

The framework has been tested with real financial data:
- **SPY (S&P 500 ETF)**: Market benchmark analysis
- **TSLA (Tesla)**: High-volatility stock analysis  
- **BND (Bond ETF)**: Fixed income analysis

Results include enhanced datasets, forecasting predictions, portfolio optimizations, and comprehensive backtesting reports.

---

**Happy Time Series Analysis! 📊⏰**

*Built with ❤️ for the data science and quantitative finance community*
