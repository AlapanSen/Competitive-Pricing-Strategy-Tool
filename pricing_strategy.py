import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pricing_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Create directories for outputs
os.makedirs('pricing_strategies', exist_ok=True)
os.makedirs('visualizations/pricing_strategies', exist_ok=True)

class PricingStrategy:
    """
    A class for implementing competitive pricing strategies
    based on the improved pricing models
    """
    
    def __init__(self, 
                models_dir='models/category_models',
                aggressive_discount_range=(0.15, 0.25),  # 15-25% below market
                min_profit_margin=0.08,  # 8% minimum profit margin
                category_min_margins=None  # Category-specific minimum margins
                ):
        """
        Initialize the pricing strategy
        
        Parameters:
        -----------
        models_dir : str
            Directory containing the trained models
        aggressive_discount_range : tuple
            Range of discount to apply (min, max) as a percentage
        min_profit_margin : float
            Default minimum profit margin to ensure
        category_min_margins : dict, optional
            Dictionary mapping category names to their minimum profit margins
        """
        self.models_dir = models_dir
        self.aggressive_discount_range = aggressive_discount_range
        self.min_profit_margin = min_profit_margin
        self.models = {}
        self.metrics = {}
        self.category_benchmarks = {}
        
        # Define category-specific cost thresholds
        # Each tuple contains (warning_threshold, viability_threshold)
        self.category_thresholds = {
            'Smartwatches': (0.95, 1.15),       # More tolerant - can handle higher costs
            'Mobile Accessories': (0.9, 1.05),   # Moderate tolerance
            'Kitchen Appliances': (0.9, 1.05),   # Moderate tolerance
            'Cameras': (0.85, 1.0),              # Lower tolerance
            'Audio': (0.8, 0.95),                # Low tolerance
            'Computers': (0.85, 1.0),            # Lower tolerance
            'Home Entertainment': (0.8, 0.95),   # Low tolerance
            'Home Improvement': (0.8, 0.95),     # Low tolerance
            'Home Office': (0.8, 0.95),          # Low tolerance 
            'Climate Control': (0.85, 1.0)       # Lower tolerance
        }
        
        # Category-specific calibration factors
        self.calibration_factors = {
            'Audio': 1.25,                # Observed 0.63x -> adjust by 1.25 to get closer to real prices
            'Cameras': 1.2,               # Observed 0.65x
            'Climate Control': 1.2,       # Observed 0.64x
            'Computers': 1.2,             # Observed 0.64x
            'Home Entertainment': 1.45,   # Observed 0.53x
            'Home Improvement': 1.2,      # Observed 0.63x
            'Home Office': 1.45,          # Observed 0.54x
            'Kitchen Appliances': 1.1,    # Observed 0.71x
            'Mobile Accessories': 1.2,    # Observed 0.69x
            'Smartwatches': 1.05          # Observed 0.77x (most accurate)
        }
        
        # Default category-specific minimum margins
        self.category_min_margins = {
            'Smartwatches': 0.05,         # Can operate on thinner margins
            'Mobile Accessories': 0.07,    # Thin margins but not as thin as smartwatches
            'Kitchen Appliances': 0.08,    # Standard margin
            'Cameras': 0.10,               # Higher margin needed
            'Audio': 0.12,                 # Higher margin needed
            'Computers': 0.09,             # Slightly higher than standard
            'Home Entertainment': 0.12,    # Higher margin needed
            'Home Improvement': 0.10,      # Higher margin needed
            'Home Office': 0.10,           # Higher margin needed
            'Climate Control': 0.09        # Slightly higher than standard
        }
        
        # Override with any provided category-specific margins
        if category_min_margins:
            self.category_min_margins.update(category_min_margins)
        
        # Load models and metrics
        self._load_models_and_metrics()
        
    def _load_models_and_metrics(self):
        """
        Load trained models and performance metrics from files
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/category_models', exist_ok=True)
        
        try:
            # Load category models
            category_models_path = 'models/category_models'
            for model_file in os.listdir(category_models_path):
                if model_file.endswith('.joblib'):
                    category = model_file.replace('_model.joblib', '')
                    model_path = os.path.join(category_models_path, model_file)
                    
                    try:
                        model_data = joblib.load(model_path)
                        self.models[category] = model_data
                        
                        # Store feature means and standard deviations for standardization
                        if 'scaler' in model_data and hasattr(model_data['scaler'], 'mean_') and hasattr(model_data['scaler'], 'scale_'):
                            self.models[category]['means'] = model_data['scaler'].mean_
                            self.models[category]['stds'] = model_data['scaler'].scale_
                        
                        logger.info(f"Loaded model for category: {category}")
                    except Exception as e:
                        logger.error(f"Error loading model for {category}: {str(e)}")
            
            # Load metrics
            metrics_path = 'models/metrics.json'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded performance metrics for {len(self.metrics)} categories")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
    
    def get_feature_extractor(self, expected_features):
        """
        Create a function that extracts features in the expected order
        from either a dictionary or pandas DataFrame
        
        Args:
            expected_features: List of feature names the model expects
            
        Returns:
            A function that extracts features in the right order
        """
        def feature_extractor(product_data):
            """
            Extract features from product data in the order expected by the model
            Works with both dictionary and DataFrame inputs
            
            Args:
                product_data: Dictionary or DataFrame containing product features
                
            Returns:
                numpy array of features in the expected order
            """
            try:
                # Handle different input types
                if isinstance(product_data, dict):
                    # Extract features from dictionary
                    features = np.array([
                        float(product_data.get(feature, 0)) 
                        for feature in expected_features
                    ]).reshape(1, -1)
                    
                elif hasattr(product_data, 'iloc'):  # Check if it's DataFrame-like
                    # Extract features from DataFrame
                    try:
                        # Try using expected_features as column names
                        features = product_data[expected_features].values.astype(float)
                    except (KeyError, TypeError):
                        # If that fails, try to use the first N columns
                        features = product_data.iloc[:, :len(expected_features)].values.astype(float)
                        
                    # Reshape if needed
                    if len(features.shape) == 1 or features.shape[0] == 1:
                        features = features.reshape(1, -1)
                        
                else:
                    # If it's already a numpy array, verify shape
                    features = np.array(product_data, dtype=float)
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)
                
                # Check if we have the right number of features
                if features.shape[1] != len(expected_features):
                    logger.warning(
                        f"Feature mismatch: Model expects {len(expected_features)} features, "
                        f"but got {features.shape[1]}. Padding with zeros."
                    )
                    # Pad with zeros if needed
                    if features.shape[1] < len(expected_features):
                        padding = np.zeros((features.shape[0], len(expected_features) - features.shape[1]))
                        features = np.hstack([features, padding])
                    else:
                        # Truncate if we have too many features
                        features = features[:, :len(expected_features)]
                
                return features
                
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
                # Return zeros as fallback
                return np.zeros((1, len(expected_features)))
                
        return feature_extractor
    
    def _safe_divide(self, a, b, default=1.0):
        """Safely divide a by b, returning default if b is 0"""
        return np.where(b > 0, a / b, default)

    def load_category_benchmarks(self, file_path='logs/outlier_stats.json'):
        """
        Load category benchmarks from various potential sources
        
        This gives us important price distribution information
        for each category to inform pricing decisions
        """
        try:
            # Try multiple paths for benchmark data
            potential_paths = [
                file_path,                                 # Original path (logs/outlier_stats.json)
                'models/category_benchmarks.json',         # Alternative path
                'models/improved/category_benchmarks.json', # Another alternative
                os.path.join(self.models_dir, 'category_benchmarks.json')  # In the models directory
            ]
            
            loaded = False
            for path in potential_paths:
                if os.path.exists(path):
                    logger.info(f"Loading benchmarks from {path}")
                    with open(path, 'r') as f:
                        benchmark_data = json.load(f)
                    loaded = True
                    break
            
            if not loaded:
                logger.warning(f"No benchmark file found in any of these locations: {potential_paths}")
                # Create empty benchmark data
                self.category_benchmarks = {category: {
                    'median_price': 1000.0,  # Default median price
                    'q1': 500.0,             # Default Q1
                    'q3': 2000.0,            # Default Q3
                    'iqr': 1500.0,           # Default IQR
                    'min_price': 100.0,      # Default min price
                    'max_price': 5000.0      # Default max price
                } for category in self.models.keys()}
                
                logger.info(f"Created default benchmarks for {len(self.category_benchmarks)} categories")
                return False
            
            # Try to extract benchmarks in different formats
            # First, check if the data is already in the expected format
            if isinstance(benchmark_data, dict) and all(
                isinstance(v, dict) and 'median_price' in v for k, v in benchmark_data.items() if isinstance(v, dict)
            ):
                # Data is already in the right format - use directly
                self.category_benchmarks = benchmark_data
                logger.info(f"Loaded benchmarks directly from file - found {len(self.category_benchmarks)} categories")
                return True
                
            # Otherwise, try to extract from outlier_stats format
            try:
                for category, stats in benchmark_data.items():
                    # Get discounted_price stats
                    if 'discounted_price' in stats:
                        price_stats = stats['discounted_price']
                        
                        q1 = float(price_stats.get('Q1', 0))
                        q3 = float(price_stats.get('Q3', 0))
                        
                        # Calculate median as average of Q1 and Q3 if not provided
                        if 'median' in price_stats:
                            median_price = float(price_stats['median'])
                        else:
                            median_price = (q1 + q3) / 2 if q1 > 0 and q3 > 0 else 1000.0
                        
                        self.category_benchmarks[category] = {
                            'median_price': median_price,
                            'q1': q1,
                            'q3': q3,
                            'iqr': float(price_stats.get('IQR', q3 - q1)),
                            'min_price': float(price_stats.get('lower_bound', q1 * 0.5)),
                            'max_price': float(price_stats.get('upper_bound', q3 * 1.5))
                        }
                    # If no discounted_price stats, try using overall stats
                    elif isinstance(stats, dict) and all(k in stats for k in ['median', 'Q1', 'Q3']):
                        median_price = float(stats.get('median', 0))
                        q1 = float(stats.get('Q1', 0))
                        q3 = float(stats.get('Q3', 0))
                        
                        self.category_benchmarks[category] = {
                            'median_price': median_price,
                            'q1': q1,
                            'q3': q3,
                            'iqr': float(stats.get('IQR', q3 - q1)),
                            'min_price': float(price_stats.get('lower_bound', q1 * 0.5)),
                            'max_price': float(price_stats.get('upper_bound', q3 * 1.5))
                        }
            except Exception as e:
                logger.error(f"Error parsing benchmark data: {str(e)}")
            
            # If no benchmarks were found, create default ones
            if not self.category_benchmarks:
                logger.warning("No category benchmarks extracted from the file")
                # Create default benchmarks for all model categories
                self.category_benchmarks = {category: {
                    'median_price': 1000.0,  # Default median price
                    'q1': 500.0,             # Default Q1
                    'q3': 2000.0,            # Default Q3
                    'iqr': 1500.0,           # Default IQR
                    'min_price': 100.0,      # Default min price
                    'max_price': 5000.0      # Default max price
                } for category in self.models.keys()}
                
                logger.info(f"Created default benchmarks for {len(self.category_benchmarks)} categories")
            else:
                logger.info(f"Loaded price benchmarks for {len(self.category_benchmarks)} categories")
            
            return True
        except Exception as e:
            logger.error(f"Error loading category benchmarks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Create default benchmark data
            self.category_benchmarks = {category: {
                'median_price': 1000.0,  # Default median price
                'q1': 500.0,             # Default Q1
                'q3': 2000.0,            # Default Q3
                'iqr': 1500.0,           # Default IQR
                'min_price': 100.0,      # Default min price
                'max_price': 5000.0      # Default max price
            } for category in self.models.keys()}
            
            logger.info(f"Created default benchmarks for {len(self.category_benchmarks)} categories due to error")
            return False

    def standardize_features(self, features, means=None, stds=None):
        """
        Standardize features without requiring original column names
        
        Args:
            features: numpy array or DataFrame of features to standardize
            means: means for standardization (if None, calculate from features)
            stds: standard deviations for standardization (if None, calculate from features)
            
        Returns:
            Tuple of (standardized_features, means, stds)
        """
        # Convert features to numpy array if it's a DataFrame
        if hasattr(features, 'values'):
            features_array = features.values
        else:
            features_array = np.array(features)
            
        # If means and stds are not provided, calculate them
        if means is None or stds is None:
            means = np.mean(features_array, axis=0)
            stds = np.std(features_array, axis=0)
            # Avoid division by zero
            stds = np.where(stds == 0, 1.0, stds)
        
        # Standardize the features
        standardized_features = (features_array - means) / stds
        
        return standardized_features, means, stds

    def predict_price(self, category, features):
        """
        Predict the product price for a given category and feature set
        
        Args:
            category: Product category
            features: Dictionary of feature values
            
        Returns:
            Predicted price or None if prediction fails
        """
        try:
            # Select the appropriate model and parameters
            model, expected_features, feature_means, feature_stds = self._select_model(category)
            
            if model is None or expected_features is None:
                logger.error(f"Cannot predict price: Missing model or feature list for category {category}")
                return None
                
            # Ensure all expected features are present
            input_features = []
            for feature in expected_features:
                if feature not in features:
                    logger.warning(f"Missing feature: {feature} for category {category}")
                    return None
                input_features.append(features[feature])
            
            # Convert to numpy array and reshape
            X = np.array(input_features).reshape(1, -1)
            
            # Apply standardization if means and stds are available
            if feature_means is not None and feature_stds is not None:
                # Standardize features
                X_std = (X - feature_means) / feature_stds
                # Handle potential division by zero in StandardScaler
                X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                X_std = X
            
            # Make prediction
            prediction = model.predict(X_std)[0]
            
            # Apply category-specific calibration if available
            if category in self.calibration_factors:
                prediction *= self.calibration_factors[category]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting price for {category}: {str(e)}")
            return None
    
    def get_competitive_price(self, features, category, market_saturation=0.5, brand_strength=0.5):
        """
        Calculate a competitive price based on features and market conditions
        
        Args:
            features: Dictionary or DataFrame with product features
            category: Product category
            market_saturation: Float [0-1] indicating market saturation (higher = more saturated)
            brand_strength: Float [0-1] indicating brand strength (higher = stronger)
            
        Returns:
            Dictionary with pricing recommendations and strategy details
        """
        try:
            # Get manufacturing cost from features
            if isinstance(features, dict):
                manufacturing_cost = features.get('manufacturing_cost', 0)
            else:
                manufacturing_cost = features['manufacturing_cost'].iloc[0] if 'manufacturing_cost' in features else 0
            
            # Get the predicted market price
            predicted_price = self.predict_price(category, features)
            if predicted_price is None:
                logger.error("Failed to predict price")
                return {"error": "Failed to predict price"}
                
            # Get margin thresholds for this category
            min_margin = self.category_min_margins.get(category, 0.10)  # Default 10% margin
            warning_threshold = self.category_thresholds.get(category, (0.6, 0.8))[0]
            viability_threshold = self.category_thresholds.get(category, (0.6, 0.8))[1]
            
            # Calculate price to cost ratio
            if manufacturing_cost > 0:
                price_to_cost_ratio = predicted_price / manufacturing_cost
            else:
                price_to_cost_ratio = float('inf')
                
            # Check viability
            viability_issue = False
            high_cost_warning = False
            
            # Handle high manufacturing cost scenario
            if price_to_cost_ratio < viability_threshold:
                viability_issue = True
                logger.warning(f"Viability issue detected for {category} product. "
                               f"Price-to-cost ratio ({price_to_cost_ratio:.2f}) below viability threshold "
                               f"({viability_threshold:.2f})")
                
                # Calculate reduced margin to keep price competitive despite high cost
                adjusted_min_margin = min_margin * 0.5  # Reduce margin to half
                logger.info(f"Reducing minimum margin from {min_margin:.2%} to {adjusted_min_margin:.2%}")
                min_margin = adjusted_min_margin
                
            elif price_to_cost_ratio < warning_threshold:
                high_cost_warning = True
                logger.warning(f"High cost warning for {category} product. "
                               f"Price-to-cost ratio ({price_to_cost_ratio:.2f}) below warning threshold "
                               f"({warning_threshold:.2f})")
                
                # Slightly reduce margin
                adjusted_min_margin = min_margin * 0.75  # Reduce margin to 75%
                logger.info(f"Reducing minimum margin from {min_margin:.2%} to {adjusted_min_margin:.2%}")
                min_margin = adjusted_min_margin
            
            # Calculate base discount range based on market saturation and brand strength
            # More saturated market -> higher discount possible
            # Stronger brand -> lower discount needed
            max_discount = 0.25  # Maximum discount of 25%
            min_discount = 0.05  # Minimum discount of 5%
            
            # Adjust discount range based on market conditions
            base_discount = max_discount * market_saturation * (1 - brand_strength * 0.5)
            
            # Ensure base discount is within reasonable limits
            discount_from_market = max(min_discount, min(base_discount, max_discount))
            
            # Calculate competitive price range
            price_min = predicted_price * (1 - discount_from_market)
            price_max = predicted_price * (1 + 0.05)  # Allow slight premium of up to 5%
            
            # Ensure price covers manufacturing cost plus minimum margin
            min_viable_price = manufacturing_cost * (1 + min_margin)
            
            # Adjust prices to ensure minimum viability
            if price_min < min_viable_price:
                price_min = min_viable_price
                
                # If even the maximum price doesn't allow minimum margin, adjust it
                if price_max < min_viable_price:
                    price_max = min_viable_price * 1.05  # 5% above minimum viable price
            
            # Calculate recommended price based on market conditions
            # In saturated markets with weak brand, aim for lower end of range
            # In less saturated markets with strong brand, aim for higher end
            position_in_range = (1 - market_saturation) * 0.6 + brand_strength * 0.4
            recommended_price = price_min + position_in_range * (price_max - price_min)
            
            # Calculate profit margin
            profit_amount = recommended_price - manufacturing_cost
            profit_margin_pct = profit_amount / recommended_price if recommended_price > 0 else 0
            
            # Calculate discount from predicted market price
            discount_pct = (predicted_price - recommended_price) / predicted_price if predicted_price > 0 else 0
            
            # Get the strategy name
            strategy_name = self._get_pricing_strategy_name(
                category, 
                market_saturation, 
                brand_strength, 
                profit_margin_pct, 
                discount_pct
            )
            
            # Package results
            result = {
                "predicted_market_price": round(predicted_price, 2),
                "price_range_min": round(price_min, 2),
                "price_range_max": round(price_max, 2),
                "recommended_price": round(recommended_price, 2),
                "discount_from_market": round(discount_pct * 100, 1),
                "profit_margin_amount": round(profit_amount, 2),
                "profit_margin_percentage": round(profit_margin_pct * 100, 1),
                "price_to_cost_ratio": round(price_to_cost_ratio, 2),
                "strategy_name": strategy_name,
                "market_conditions": "Saturated" if market_saturation > 0.7 else 
                                    "Moderate" if market_saturation > 0.3 else "Emerging",
                "brand_position": "Strong" if brand_strength > 0.7 else 
                                 "Moderate" if brand_strength > 0.3 else "Weak",
                "viability_issue": viability_issue,
                "high_cost_warning": high_cost_warning
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in competitive pricing: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _calculate_percentile_position(self, price, q1, q3):
        """Calculate approximate percentile position based on quartile range"""
        if price <= q1:
            return 25 * (price / q1)
        elif price <= q3:
            return 25 + (50 * (price - q1) / (q3 - q1))
        else:
            return 75 + (25 * min((price - q3) / (q3 - q1), 1.0))
        
    def _get_pricing_strategy_name(self, market_saturation, brand_strength, 
                                  profit_margin, discount_percentage, category=None):
        """
        Get a descriptive name for the pricing strategy
        
        Parameters:
        -----------
        market_saturation : str
            'low', 'medium', or 'high'
        brand_strength : str
            'low', 'medium', or 'high'
        profit_margin : float
            Profit margin as a decimal
        discount_percentage : float
            Discount from market price as a percentage
        category : str, optional
            Product category for category-specific strategies
            
        Returns:
        --------
        str
            Strategy name
        """
        
        # Category-specific strategies
        if category:
            # Smartwatches can support higher margins
            if category == 'Smartwatches' and profit_margin > 0.35:
                return "Premium Brand Position"
            
            # Mobile accessories benefit from volume
            if category == 'Mobile Accessories' and discount_percentage > 20:
                return "High-Volume Entry Strategy"
                
            # Audio requires quality perception
            if category == 'Audio' and profit_margin > 0.3 and brand_strength == 'high':
                return "Premium Audio Experience"
                
            # Computers benefit from feature emphasis
            if category == 'Computers' and profit_margin < 0.15:
                return "Feature-Value Balance Strategy"
        
        # Base strategy on discount and profit margin
        if discount_percentage >= 25:
            base_strategy = "Aggressive Market Entry"
        elif discount_percentage >= 15:
            base_strategy = "Value-Oriented Strategy"
        elif discount_percentage >= 5:
            base_strategy = "Competitive Positioning"
        else:
            base_strategy = "Premium Positioning"
            
        # Consider market conditions
        if market_saturation == "high" and discount_percentage >= 20:
            return "Deep Discount Strategy"
        elif market_saturation == "high" and brand_strength == "low":
            return "Undercut Competitors Strategy"
        elif brand_strength == "high" and discount_percentage < 10:
            return "Brand Premium Strategy"
        elif profit_margin < 0.15:  # Less than 15% margin
            return "Thin-Margin Volume Strategy"
        else:
            return base_strategy
    
    def visualize_pricing_recommendation(self, recommendation, save_path=None):
        """
        Visualize the pricing recommendation with a chart
        
        Parameters:
        -----------
        recommendation : dict
            Pricing recommendation from get_competitive_price
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The visualization figure
        """
        try:
            category = recommendation['category']
            market_price = recommendation['predicted_market_price']
            recommended_price = recommendation['recommended_price']
            manufacturing_cost = recommendation['manufacturing_cost']
            min_competitive_price = recommendation['min_competitive_price']
            max_competitive_price = recommendation['max_competitive_price']
            minimum_viable_price = recommendation['minimum_viable_price']
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(12, 8))
            spec = fig.add_gridspec(2, 1, height_ratios=[2, 1])
            
            # Pricing strategy visualization (top)
            ax1 = fig.add_subplot(spec[0])
            
            # Plot price points
            price_points = [
                {'name': 'Manufacturing Cost', 'price': manufacturing_cost, 'color': 'gray'},
                {'name': 'Minimum Viable Price', 'price': minimum_viable_price, 'color': 'orange'},
                {'name': 'Min Competitive', 'price': min_competitive_price, 'color': 'green'},
                {'name': 'Recommended Price', 'price': recommended_price, 'color': 'blue'},
                {'name': 'Max Competitive', 'price': max_competitive_price, 'color': 'green'},
                {'name': 'Market Price', 'price': market_price, 'color': 'red'}
            ]
            
            # Sort by price
            price_points.sort(key=lambda x: x['price'])
            
            # Create y positions
            y_positions = np.linspace(0.2, 0.8, len(price_points))
            
            # Plot horizontal price line
            min_price = min(p['price'] for p in price_points) * 0.9
            max_price = max(p['price'] for p in price_points) * 1.1
            ax1.axhline(y=0.5, xmin=0, xmax=1, color='black', alpha=0.2, linestyle='--')
            
            # Plot price points
            for i, point in enumerate(price_points):
                y_pos = y_positions[i]
                ax1.scatter(point['price'], y_pos, color=point['color'], s=100, zorder=3)
                ax1.text(point['price'], y_pos+0.05, f"₹{point['price']:.2f}", 
                        ha='center', va='bottom', fontweight='bold')
                ax1.text(point['price'], y_pos-0.05, point['name'], 
                        ha='center', va='top')
            
            # Highlight the competitive range
            ax1.axvspan(min_competitive_price, max_competitive_price, alpha=0.2, color='green', zorder=1)
            ax1.text((min_competitive_price + max_competitive_price)/2, 0.1, 
                    'Competitive Range', ha='center', fontweight='bold')
            
            # Formatting
            ax1.set_xlim(min_price, max_price)
            ax1.set_ylim(0, 1)
            ax1.set_title(f'Pricing Strategy for {category}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Price (₹)', fontsize=12)
            ax1.xaxis.set_ticks_position('bottom')
            ax1.yaxis.set_visible(False)
            
            # Add profit margin and discount info
            profit_margin = recommendation['profit_margin_percentage']
            discount = recommendation['discount_from_market']
            strategy = recommendation['strategy_name']
            
            ax1.text(min_price + (max_price - min_price) * 0.02, 0.95, 
                    f"Strategy: {strategy}", fontsize=12, fontweight='bold')
            ax1.text(min_price + (max_price - min_price) * 0.02, 0.90, 
                    f"Profit Margin: {profit_margin:.1f}%", fontsize=12)
            ax1.text(min_price + (max_price - min_price) * 0.02, 0.85, 
                    f"Discount from Market: {discount:.1f}%", fontsize=12)
            
            # Price sensitivity curve (bottom)
            ax2 = fig.add_subplot(spec[1])
            
            # Get estimated elasticity
            elasticity = recommendation.get('price_elasticity', -1.5)
            
            # Create a range of discount percentages
            discount_range = np.linspace(0, 50, 100)
            
            # Calculate sales impact for each discount
            sales_impact = -elasticity * discount_range
            
            # Calculate net revenue impact (simplified model)
            # Revenue impact = (1 + sales_impact) * (1 - discount/100) - 1
            revenue_impact = (1 + sales_impact/100) * (1 - discount_range/100) - 1
            revenue_impact = revenue_impact * 100  # Convert to percentage
            
            # Plot the curves
            ax2.plot(discount_range, sales_impact, label='Sales Volume Impact', 
                    color='blue', linewidth=2)
            ax2.plot(discount_range, revenue_impact, label='Revenue Impact', 
                    color='green', linewidth=2)
            
            # Mark the recommended discount
            ax2.axvline(x=discount, color='red', linestyle='--', alpha=0.7)
            ax2.text(discount + 1, -5, f'{discount:.1f}% Discount', 
                    color='red', fontweight='bold')
            
            # Formatting
            ax2.set_xlim(0, 50)
            ax2.set_ylim(-40, 80)
            ax2.set_title('Price Sensitivity Analysis', fontsize=12)
            ax2.set_xlabel('Discount from Market Price (%)', fontsize=10)
            ax2.set_ylabel('Impact (%)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Add annotations for the estimated impact
            sales_impact_value = recommendation.get('estimated_sales_impact', 0)
            ax2.annotate(f'+{sales_impact_value:.1f}% Sales', 
                        xy=(discount, sales_impact_value), 
                        xytext=(discount+5, sales_impact_value+10),
                        arrowprops=dict(arrowstyle='->', color='blue'))
            
            # Annotate the revenue impact at the recommended discount
            revenue_impact_at_discount = (1 + sales_impact_value/100) * (1 - discount/100) - 1
            revenue_impact_at_discount = revenue_impact_at_discount * 100
            ax2.annotate(f'{revenue_impact_at_discount:.1f}% Revenue', 
                        xy=(discount, revenue_impact_at_discount), 
                        xytext=(discount+5, revenue_impact_at_discount-10),
                        arrowprops=dict(arrowstyle='->', color='green'))
            
            # Use figure.subplots_adjust instead of tight_layout to avoid warnings
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3)
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing pricing recommendation: {str(e)}")
            return None
        finally:
            # Ensure the figure is closed to prevent memory leaks
            if 'fig' in locals():
                plt.close(fig)

    def _select_model(self, category):
        """
        Select the appropriate model and standardization parameters for a given product category
        
        Args:
            category: Product category (str)
            
        Returns:
            Tuple of (model, feature_names, feature_means, feature_stds)
        """
        try:
            # Check if model exists for this category
            if category not in self.models:
                logger.warning(f"No model found for category: {category}")
                return None, None, None, None
            
            model_data = self.models.get(category, {})
            
            # Extract model and standardization parameters
            model = model_data.get('model')
            feature_means = model_data.get('means')
            feature_stds = model_data.get('stds')
            expected_features = model_data.get('expected_features')
            
            if model is None:
                logger.warning(f"Model object not found for category: {category}")
                return None, None, None, None
                
            return model, expected_features, feature_means, feature_stds
            
        except Exception as e:
            logger.error(f"Error selecting model for category {category}: {str(e)}")
            return None, None, None, None

def test_pricing_strategy():
    """Test function to demonstrate the pricing strategy"""
    try:
        logger.info("Testing pricing strategy implementation...")
        
        # Initialize strategy
        strategy = PricingStrategy()
        
        # Load benchmarks
        benchmark_loaded = strategy.load_category_benchmarks()
        logger.info(f"Benchmark loading {'successful' if benchmark_loaded else 'failed'}")
        
        # Test with a sample product
        sample_categories = list(strategy.models.keys())
        
        if not sample_categories:
            logger.error("No models available for testing")
            return
        
        logger.info(f"Available categories: {sample_categories}")
        sample_category = sample_categories[0]
        
        logger.info(f"Testing with sample category: {sample_category}")
        
        # Get model data to see what features it expects
        if sample_category in strategy.models:
            model_data = strategy.models[sample_category]
            if hasattr(model_data['model'], 'feature_names_in_'):
                logger.info(f"Model expects these features: {model_data['model'].feature_names_in_}")
        
        # Sample product features
        # Include more features to match what the model expects
        features = {
            'rating': 4.2,
            'rating_count': 120,
            'discount_percentage': 15,
            'manufacturing_cost': 100,
            'price_to_cost_ratio': 2.5,
            'margin_percentage': 40,
            'brand_strength_score': 0.6,
            'production_cost': 100,  # Same as manufacturing_cost
            'quality_score': 80      # Rating * 20
        }
        
        logger.info(f"Testing with features: {features}")
        
        # Test prediction
        prediction = strategy.predict_price(sample_category, features)
        
        if prediction:
            logger.info(f"Prediction successful: {prediction}")
            logger.info(f"Predicted market price: ₹{prediction:.2f}")
            
            # Test recommendations with different scenarios
            scenarios = [
                {'name': 'New Brand, High Competition', 'saturation': 'high', 'strength': 'low'},
                {'name': 'Average Brand, Average Competition', 'saturation': 'medium', 'strength': 'medium'},
                {'name': 'Strong Brand, Low Competition', 'saturation': 'low', 'strength': 'high'}
            ]
            
            for scenario in scenarios:
                try:
                    logger.info(f"Testing scenario: {scenario['name']}")
                    recommendation = strategy.get_competitive_price(
                        features,
                        sample_category,
                        scenario['saturation'],
                        scenario['strength']
                    )
                    
                    if recommendation:
                        logger.info(f"Scenario: {scenario['name']}")
                        logger.info(f"Recommended price: ₹{recommendation['recommended_price']:.2f}")
                        logger.info(f"Discount from market: {recommendation['discount_from_market']:.1f}%")
                        logger.info(f"Profit margin: {recommendation['profit_margin_percentage']:.1f}%")
                        logger.info(f"Strategy: {recommendation['strategy_name']}")
                        
                        # Save visualization
                        try:
                            fig = strategy.visualize_pricing_recommendation(recommendation)
                            if fig:
                                save_path = os.path.join(
                                    'visualizations/pricing_strategies',
                                    f"{sample_category}_{scenario['saturation']}_{scenario['strength']}.png"
                                )
                                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                                plt.close(fig)
                                logger.info(f"Saved visualization to {save_path}")
                        except Exception as e:
                            logger.error(f"Error saving visualization: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in scenario {scenario['name']}: {str(e)}")
        else:
            logger.error("Failed to generate prediction")
            
    except Exception as e:
        logger.error(f"Error testing pricing strategy: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Test the implementation
    # First make sure directories exist
    os.makedirs('pricing_strategies', exist_ok=True)
    os.makedirs('visualizations/pricing_strategies', exist_ok=True)
    
    # Run the test
    test_pricing_strategy() 