# Competitive Pricing Strategy System

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.26.0-red.svg)
![XGBoost](https://img.shields.io/badge/xgboost-1.7.6-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

A machine learning-based system that helps new sellers enter the market with optimized pricing strategies tailored to specific product categories.

## 🌟 Features

- **Category-specific pricing models**: Tailored price predictions for each product category
- **Competitive pricing algorithms**: Multiple strategies based on market conditions
- **Interactive Streamlit dashboard**: User-friendly interface for price recommendations
- **Comprehensive analytics**: Performance visualization and market analysis tools
- **Category-specific calibration**: Custom profit margins and viability thresholds

## 📋 Project Structure

```
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed data files
│   └── engineered/                # Feature engineered data
├── models/                        # Trained models
│   ├── category_models/           # Original category models
│   └── improved/                  # Improved models with better performance
├── visualizations/                # Visualizations
│   ├── distributions/             # Price distributions
│   ├── model_performance/         # Model performance charts
│   ├── feature_importance/        # Feature importance visualizations
│   └── pricing_strategies/        # Pricing strategy visualizations
├── logs/                          # Log files
├── pricing_strategies/            # Saved pricing recommendations
├── data_preparation.py            # Data preparation script
├── feature_engineering_fixed.py   # Feature engineering script
├── model_development.py           # Original model development
├── improved_model_development.py  # Improved model development
├── pricing_strategy.py            # Pricing strategy implementation
├── streamlit_app.py               # Streamlit application
├── test_pricing_model.py          # Testing script for pricing models
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/competitive-pricing-strategy.git
   cd competitive-pricing-strategy
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Usage

### Data Preparation

Process the raw data and prepare it for model training:

```bash
python data_preparation.py
```

### Feature Engineering

Generate features for the machine learning models:

```bash
python feature_engineering_fixed.py
```

### Model Training

Train the improved XGBoost models for each product category:

```bash
python improved_model_development.py
```

### Pricing Strategy Testing

Test the pricing strategy implementation:

```bash
python test_pricing_model.py
```

### Launch the Streamlit App

Start the interactive dashboard for pricing recommendations:

```bash
streamlit run streamlit_app.py
```

## 📊 Pricing Strategies

The system implements several pricing strategies based on market conditions:

1. **Category-specific strategies**:
   - **Smartwatches**: Premium positioning for high-end models, aggressive entry for budget models
   - **Mobile Accessories**: Volume-based strategies with thin margins
   - **Audio**: Quality-focused pricing with strong brand positioning
   - **Computers**: Feature-driven pricing with technical specification consideration

2. **Market condition strategies**:
   - **Aggressive Market Entry**: 25%+ discount for quick market entry
   - **Value-Oriented Strategy**: 15-25% discount balancing value and profit
   - **Competitive Positioning**: 5-15% discount for established products
   - **Premium Positioning**: <5% discount for premium products
   - **Deep Discount Strategy**: For highly saturated markets
   - **Thin-Margin Volume Strategy**: For price-sensitive categories

## 📈 Model Performance

Our models are evaluated using multiple metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Within-10% Percentage (% of predictions within 10% of actual)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or feedback, please open an issue in the GitHub repository. 