# Quick Start Guide

This guide will help you get the Competitive Pricing Strategy system up and running quickly.

## Option 1: Using Docker (Recommended)

The fastest way to get started is using Docker:

1. Make sure you have [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/) installed

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/competitive-pricing-strategy.git
   cd competitive-pricing-strategy
   ```

3. Start the application:
   ```bash
   docker-compose up -d
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Option 2: Manual Setup

If you prefer not to use Docker:

1. Make sure you have Python 3.10+ installed

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/competitive-pricing-strategy.git
   cd competitive-pricing-strategy
   ```

3. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Process data and train models (optional, if you don't have pre-trained models):
   ```bash
   python data_preparation.py
   python feature_engineering_fixed.py
   python improved_model_development.py
   ```

6. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

7. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Using the Application

1. **Single Product Pricing**:
   - Select a product category
   - Enter manufacturing cost and other details
   - Click "Generate Pricing Strategy"

2. **Batch Pricing**:
   - Upload a CSV file with product details
   - Download recommendations

3. **Market Analysis**:
   - Explore pricing trends for different categories
   - View competitive landscape analysis

## Sample Data

To test the system, sample data is provided in the `data/examples` directory. 