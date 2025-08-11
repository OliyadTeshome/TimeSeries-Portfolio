# TimeSeries-Portfolio

A comprehensive time series analysis and portfolio optimization framework built with Python. This project provides tools for data preprocessing, exploratory data analysis, forecasting, portfolio optimization, and backtesting.

## 🚀 Features

### Data Management
- **Data Loading**: Support for multiple data sources (CSV, APIs, databases)
- **Data Preprocessing**: Cleaning, normalization, feature engineering
- **Data Validation**: Comprehensive data quality checks

### Time Series Analysis
- **Exploratory Data Analysis**: Statistical summaries, visualizations, pattern detection
- **Stationarity Testing**: ADF, KPSS tests with automatic differencing
- **Seasonal Decomposition**: Trend, seasonal, and residual analysis

### Forecasting Models
- **ARIMA Models**: Auto-regressive integrated moving average with automatic parameter selection
- **LSTM Models**: Long short-term memory neural networks for complex patterns
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Model Evaluation**: Comprehensive performance metrics and diagnostics

### Portfolio Optimization
- **Risk-Return Optimization**: Maximum Sharpe ratio, minimum variance portfolios
- **Efficient Frontier**: Generate optimal portfolio combinations
- **Asset Allocation**: Equal weight, value-weighted, and maximum diversification strategies
- **Risk Metrics**: VaR, CVaR, maximum drawdown analysis

### Backtesting Framework
- **Strategy Testing**: Buy & hold, momentum, mean reversion strategies
- **Performance Analysis**: Returns, risk metrics, drawdown analysis
- **Transaction Costs**: Realistic trading simulation with costs and slippage
- **Strategy Comparison**: Multi-strategy performance evaluation

## 📁 Project Structure

```
TimeSeries-Portfolio/
│
├── data/                          # Raw and processed data
│   ├── raw/                      # Original data files
│   └── processed/                # Cleaned and transformed data
│
├── notebooks/                     # Jupyter Notebooks for each task
│   ├── task1_preprocessing_eda.ipynb      # Data preprocessing and EDA
│   ├── task2_forecasting_models.ipynb     # ARIMA and LSTM models
│   ├── task3_future_forecast_analysis.ipynb # Future forecasting analysis
│   ├── task4_portfolio_optimization.ipynb  # Portfolio optimization
│   └── task5_backtesting.ipynb            # Strategy backtesting
│
├── src/                           # Python modules for reusable code
│   ├── data_loader.py            # Data loading and management
│   ├── preprocessing.py          # Data cleaning and transformation
│   ├── eda.py                    # Exploratory data analysis
│   ├── arima_model.py            # ARIMA model implementation
│   ├── lstm_model.py             # LSTM model implementation
│   ├── forecasting.py            # Unified forecasting interface
│   ├── portfolio_optimization.py # Portfolio optimization tools
│   └── backtesting.py            # Backtesting framework
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

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/TimeSeries-Portfolio.git
   cd TimeSeries-Portfolio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, matplotlib, seaborn; print('Installation successful!')"
   ```

## 📊 Quick Start

### 1. Data Preprocessing and EDA
```python
from src.data_loader import DataLoader
from src.preprocessing import preprocess_pipeline
from src.eda import run_complete_eda

# Load data
loader = DataLoader()
data = loader.load_csv('your_data.csv')

# Preprocess data
processed_data, summary = preprocess_pipeline(
    data,
    handle_missing=True,
    remove_outliers=True,
    normalize=True,
    create_lags=True
)

# Run EDA
eda_results = run_complete_eda(processed_data)
```

### 2. Time Series Forecasting
```python
from src.forecasting import run_complete_forecasting

# Run complete forecasting pipeline
forecasting_engine = run_complete_forecasting(
    data['value'], 
    forecast_steps=30
)

# Access results
forecasts = forecasting_engine.forecasts
ensemble_forecast = forecasting_engine.ensemble_forecast
```

### 3. Portfolio Optimization
```python
from src.portfolio_optimization import run_portfolio_optimization

# Create returns data
returns_data = data.pct_change().dropna()

# Run portfolio optimization
optimizer = run_portfolio_optimization(
    returns_data, 
    risk_free_rate=0.02
)

# Access results
optimal_weights = optimizer.optimal_weights
efficient_frontier = optimizer.efficient_frontier
```

### 4. Strategy Backtesting
```python
from src.backtesting import run_complete_backtesting

# Run backtesting
backtester = run_complete_backtesting(
    market_data, 
    initial_capital=100000
)

# Compare strategies
comparison = backtester.compare_strategies()
best_strategy = backtester.get_best_strategy('sharpe_ratio')
```

## 📚 Jupyter Notebooks

The project includes comprehensive Jupyter notebooks that walk through each analysis step:

1. **Task 1: Preprocessing & EDA** - Data cleaning, exploration, and visualization
2. **Task 2: Forecasting Models** - ARIMA and LSTM model implementation
3. **Task 3: Future Forecasts** - Long-term forecasting and analysis
4. **Task 4: Portfolio Optimization** - Risk-return optimization strategies
5. **Task 5: Backtesting** - Strategy evaluation and performance analysis

## 🔧 Configuration

### Data Sources
- Update data paths in `src/data_loader.py`
- Configure API keys for external data sources
- Set data format preferences

### Model Parameters
- Adjust ARIMA parameter ranges in `src/arima_model.py`
- Modify LSTM architecture in `src/lstm_model.py`
- Configure optimization constraints in `src/portfolio_optimization.py`

### Backtesting Settings
- Set transaction costs and slippage in `src/backtesting.py`
- Configure strategy parameters
- Adjust performance metrics thresholds

## 📈 Output and Reports

The framework generates comprehensive outputs:

- **Visualizations**: Time series plots, correlation matrices, performance charts
- **Reports**: Markdown reports with analysis summaries and recommendations
- **Data Files**: Processed datasets, model results, optimization outputs
- **Models**: Saved trained models for future use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with popular Python data science libraries
- Inspired by financial time series analysis best practices
- Designed for educational and research purposes

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the notebooks
- Review the source code examples

## 🔮 Future Enhancements

- Additional forecasting models (Prophet, NeuralProphet)
- Real-time data streaming capabilities
- Web dashboard interface
- Advanced risk management tools
- Multi-asset class support

---

**Happy Time Series Analysis! 📊⏰**
