# Model Files

This directory should contain trained models for the Competitive Pricing Strategy application.

## Required Files

For the application to work properly, the following files/directories should be present:

### Category Models
- `improved/category_models/`: Directory containing trained models for each product category
- `improved/category_benchmarks.json`: JSON file with price benchmarks for each category

## Important Note for Deployment

When deploying to Streamlit Cloud, you need to include at least the category benchmark file (`improved/category_benchmarks.json`).

If you don't have trained models, the application will attempt to create default benchmarks, but this can lead to errors or inaccurate pricing recommendations. For a full deployment, you should run the model training scripts locally and push the resulting model files to your repository.

## Instructions for Generating Models

To generate model files:
1. Run `data_preparation.py` to prepare the dataset
2. Run `feature_engineering_fixed.py` to extract features
3. Run `improved_model_development.py` to train the models

This will generate all necessary model files in the appropriate directories. 