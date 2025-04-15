FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better cache utilization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Create directories for data and models if they don't exist
RUN mkdir -p data/raw data/processed data/engineered models/category_models models/improved logs visualizations pricing_strategies

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 