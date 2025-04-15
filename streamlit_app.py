import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import base64

# Import our pricing strategy implementation
sys.path.append('.')
from pricing_strategy import PricingStrategy

# Create necessary directories
os.makedirs('pricing_strategies', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('visualizations/pricing_strategies', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Competitive Pricing Strategy Tool",
    page_icon="💰",
    layout="wide"
)

# Workaround function for file download
def get_download_link(df, filename, text):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

# Initialize session state
if 'pricing_strategy' not in st.session_state:
    # Explicitly specify the models directory path for Streamlit Cloud compatibility
    models_dir = os.path.join(os.getcwd(), 'models', 'improved', 'category_models')
    st.session_state.pricing_strategy = PricingStrategy(models_dir=models_dir)
    # Load benchmarks
    st.session_state.pricing_strategy.load_category_benchmarks()
    
    # Debug info about benchmarks
    benchmark_info = {
        "Models Directory": models_dir,
        "Benchmark Categories": list(st.session_state.pricing_strategy.category_benchmarks.keys()),
        "Total Categories": len(st.session_state.pricing_strategy.category_benchmarks),
        "Sample Data": str(list(st.session_state.pricing_strategy.category_benchmarks.values())[:2])
    }
    st.sidebar.expander("Debug: Benchmark Data", expanded=False).write(benchmark_info)

# Sidebar
# Using a hosted image instead of local
st.sidebar.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=100)
st.sidebar.title("Pricing Strategy Tool")
st.sidebar.markdown("### For New Sellers")
st.sidebar.markdown("This tool helps new sellers enter the market with competitive pricing strategies.")

# Get available categories
available_categories = list(st.session_state.pricing_strategy.models.keys())

if not available_categories:
    # When no models are available, create default categories for demo
    default_categories = [
        'Audio', 'Cameras', 'Climate Control', 'Computers', 
        'Home Entertainment', 'Home Improvement', 'Home Office',
        'Kitchen Appliances', 'Mobile Accessories', 'Smartwatches'
    ]
    
    st.warning("""
    ⚠️ No trained models found. Running in demo mode with simulated models.
    You can still explore the interface and functionality.
    """)
    
    # Create a simulated pricing strategy
    class SimulatedPricingStrategy(PricingStrategy):
        def predict_price(self, product_features, category):
            """Simulate a price prediction"""
            base_price = product_features['manufacturing_cost'] * 2.5
            if 'rating' in product_features:
                base_price *= (1 + 0.1 * (product_features['rating'] - 3))
            
            return {
                'predicted_market_price': base_price,
                'confidence': 0.8,
                'category': category
            }
    
    # Replace with simulated strategy
    st.session_state.pricing_strategy = SimulatedPricingStrategy()
    available_categories = default_categories
    
    # Add simulated models for UI to work
    for category in available_categories:
        st.session_state.pricing_strategy.models[category] = {'model': None, 'scaler': None}

# Main content
st.title("Competitive Pricing Strategy Tool")

# Add notification about model improvements
st.success("""
🚀 **Model Improvements**: We've enhanced the pricing engine with category-specific calibration, custom profit margins, 
and adaptive viability thresholds. These improvements make our recommendations more precise for each product category.
See the "Category Settings" tab for details.
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Product Pricing", "Batch Pricing", "Market Analysis", "Pricing Strategy History", "Category Settings"])

# Tab 1: Single Product Pricing
with tab1:
    st.header("Generate Pricing Recommendation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Basic product information
        category = st.selectbox("Product Category", available_categories)
        product_name = st.text_input("Product Name", "New Product")
        manufacturing_cost = st.number_input("Manufacturing Cost (₹)", min_value=1.0, value=100.0, step=10.0)
        
        # Display category-specific settings for the selected category
        if category in st.session_state.pricing_strategy.calibration_factors:
            with st.expander("Category-Specific Settings", expanded=False):
                settings_cols = st.columns(3)
                
                with settings_cols[0]:
                    calibration = st.session_state.pricing_strategy.calibration_factors.get(category, 1.0)
                    st.metric("Calibration Factor", f"{calibration:.2f}x")
                    st.caption("Adjusts price predictions for this category")
                
                with settings_cols[1]:
                    min_margin = st.session_state.pricing_strategy.category_min_margins.get(category, 0.1) * 100
                    st.metric("Min Profit Margin", f"{min_margin:.1f}%")
                    st.caption("Lowest acceptable profit margin")
                
                with settings_cols[2]:
                    warning, viability = st.session_state.pricing_strategy.category_thresholds.get(category, (0.85, 1.0))
                    st.metric("Warning Threshold", f"{warning:.2f}x")
                    st.metric("Viability Threshold", f"{viability:.2f}x")
                    st.caption("Cost-to-price ratio thresholds")
        
        # Market conditions
        st.subheader("Market Conditions")
        market_saturation = st.select_slider(
            "Market Saturation", 
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Low: Few competitors, High: Many competitors"
        )
        
        brand_strength = st.select_slider(
            "Brand Strength",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Low: New brand, High: Well-established brand"
        )
    
    with col2:
        # Product features
        st.subheader("Product Features")
        
        rating = st.slider("Product Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        rating_count = st.number_input("Number of Ratings", min_value=1, value=100, step=10)
        discount_percentage = st.slider("Current Market Discount %", min_value=0, max_value=70, value=15)
        
        # Advanced features (optional)
        with st.expander("Advanced Features"):
            price_to_cost_ratio = st.slider(
                "Industry Average Price-to-Cost Ratio", 
                min_value=1.1, 
                max_value=5.0, 
                value=2.5,
                step=0.1,
                help="Higher values mean higher markup is common in this category"
            )
            margin_percentage = st.slider(
                "Industry Average Margin %", 
                min_value=5, 
                max_value=80, 
                value=40,
                help="The average profit margin percentage in this category"
            )
    
    # Create features dictionary
    features = {
        'rating': rating,
        'rating_count': rating_count,
        'discount_percentage': discount_percentage,
        'manufacturing_cost': manufacturing_cost,
        'price_to_cost_ratio': price_to_cost_ratio,
        'margin_percentage': margin_percentage,
        'log_manufacturing_cost': np.log1p(manufacturing_cost)
    }
    
    # Button to generate pricing
    if st.button("Generate Pricing Strategy"):
        with st.spinner("Analyzing market conditions and generating recommendation..."):
            # Make prediction
            prediction = st.session_state.pricing_strategy.predict_price(features, category)
            
            if prediction:
                # Get recommendation
                recommendation = st.session_state.pricing_strategy.get_competitive_price(
                    prediction,
                    manufacturing_cost,
                    market_saturation.lower(),
                    brand_strength.lower()
                )
                
                if recommendation:
                    # Display results
                    st.session_state.recommendation = recommendation
                    
                    # Display the visualization
                    fig = st.session_state.pricing_strategy.visualize_pricing_recommendation(recommendation)
                    if fig:
                        st.pyplot(fig)
                        
                        # Save recommendation
                        filename = f"{product_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        filepath = os.path.join('pricing_strategies', f"{filename.replace(' ', '_')}.json")
                        os.makedirs('pricing_strategies', exist_ok=True)
                        
                        try:
                            with open(filepath, 'w') as f:
                                json.dump(recommendation, f, indent=4)
                                
                            st.success(f"Pricing recommendation saved to {filepath}")
                        except Exception as e:
                            st.warning(f"Unable to save recommendation to file: {str(e)}")
                        
                        # Display the actual recommendation values in a cleaner format
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Market Price", f"₹{recommendation['predicted_market_price']:.2f}")
                            st.metric("Manufacturing Cost", f"₹{recommendation['manufacturing_cost']:.2f}")
                        
                        with col2:
                            st.metric("Recommended Price", f"₹{recommendation['recommended_price']:.2f}")
                            st.metric("Discount from Market", f"{recommendation['discount_from_market']:.1f}%")
                        
                        with col3:
                            st.metric("Profit Margin", f"{recommendation['profit_margin_percentage']:.1f}%")
                            st.metric("Strategy", recommendation['strategy'])
                        
                        # Check for viability issues
                        if 'viability_issue' in recommendation and recommendation['viability_issue']:
                            st.error("""
                            ### ⚠️ Product Viability Issue
                            The manufacturing cost is too high relative to the market price, 
                            making it difficult to price competitively while maintaining minimum profit margins.
                            """)

# Tab 2: Batch Pricing
with tab2:
    st.header("Batch Pricing")
    
    st.info("""
    📊 **Enhanced Batch Pricing**: Recommendations use category-specific calibration factors, minimum profit margins,
    and viability thresholds for more accurate results. Each product category has custom settings optimized for its market dynamics.
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file with products to price", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Check if required columns exist
            required_columns = ['category', 'product_name', 'manufacturing_cost']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Show expected format
                st.markdown("""
                ## Expected CSV Format:
                Your CSV should have the following columns:
                - `category`: The product category (must match available models)
                - `product_name`: Name of the product
                - `manufacturing_cost`: Manufacturing cost in dollars
                
                Optional columns:
                - `rating`: Product rating (1-5)
                - `rating_count`: Number of ratings
                - `discount_percentage`: Market discount percentage
                - `price_to_cost_ratio`: Industry average price to cost ratio
                - `margin_percentage`: Industry average margin percentage
                - `market_saturation`: "Low", "Medium", or "High"
                - `brand_strength`: "Low", "Medium", or "High"
                """)
                
                # Show sample format
                sample_data = {
                    'category': ['Computers', 'Audio', 'Mobile Accessories'],
                    'product_name': ['New Laptop', 'Wireless Earbuds', 'Phone Case'],
                    'manufacturing_cost': [450, 40, 5],
                    'rating': [4.2, 4.5, 4.0],
                    'rating_count': [120, 350, 980],
                    'discount_percentage': [12, 15, 25],
                    'market_saturation': ['Medium', 'High', 'High'],
                    'brand_strength': ['Low', 'Medium', 'Low']
                }
                
                sample_df = pd.DataFrame(sample_data)
                
                st.write("Sample format:")
                st.dataframe(sample_df)
                
                # Add download sample button
                csv = sample_df.to_csv(index=False)
                st.markdown(get_download_link(sample_df, "sample_products.csv", "Download Sample CSV"), unsafe_allow_html=True)
                
            else:
                # Process the batch
                st.subheader("Batch Pricing Results")
                
                # Check if we have valid categories
                invalid_categories = [c for c in df['category'].unique() 
                                     if c not in available_categories]
                
                if invalid_categories:
                    st.warning(f"""
                    Some products have categories that don't match our models: {', '.join(invalid_categories)}
                    Please ensure all categories match one of: {', '.join(available_categories)}
                    """)
                
                # Use default values for missing optional columns
                if 'rating' not in df.columns:
                    df['rating'] = 4.0
                if 'rating_count' not in df.columns:
                    df['rating_count'] = 100
                if 'discount_percentage' not in df.columns:
                    df['discount_percentage'] = 15
                if 'price_to_cost_ratio' not in df.columns:
                    df['price_to_cost_ratio'] = 2.5
                if 'margin_percentage' not in df.columns:
                    df['margin_percentage'] = 40
                if 'market_saturation' not in df.columns:
                    df['market_saturation'] = 'Medium'
                if 'brand_strength' not in df.columns:
                    df['brand_strength'] = 'Medium'
                
                # Process each product
                results = []
                
                progress_bar = st.progress(0)
                for i, row in enumerate(df.iterrows()):
                    index, product = row
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(df))
                    
                    # Get prediction for valid categories only
                    if product['category'] in available_categories:
                        # Prepare features
                        features = {
                            'rating': product['rating'],
                            'rating_count': product['rating_count'],
                            'discount_percentage': product['discount_percentage'],
                            'manufacturing_cost': product['manufacturing_cost'],
                            'price_to_cost_ratio': product['price_to_cost_ratio'],
                            'margin_percentage': product['margin_percentage'],
                            'log_manufacturing_cost': np.log1p(product['manufacturing_cost'])
                        }
                        
                        # Get prediction
                        prediction = st.session_state.pricing_strategy.predict_price(
                            features, product['category'])
                        
                        if prediction:
                            # Get recommendation
                            market_sat = str(product['market_saturation']).lower()
                            brand_str = str(product['brand_strength']).lower()
                            
                            recommendation = st.session_state.pricing_strategy.get_competitive_price(
                                prediction, product['manufacturing_cost'], market_sat, brand_str)
                            
                            if recommendation:
                                # Add to results
                                result = {
                                    'product_name': product['product_name'],
                                    'category': product['category'],
                                    'manufacturing_cost': product['manufacturing_cost'],
                                    'predicted_market_price': recommendation['predicted_market_price'],
                                    'recommended_price': recommendation['recommended_price'],
                                    'discount_from_market': recommendation['discount_from_market'],
                                    'profit_margin_percentage': recommendation['profit_margin_percentage'],
                                    'strategy': recommendation['strategy']
                                }
                                
                                # Add viability warning flag if present
                                if 'viability_issue' in recommendation and recommendation['viability_issue']:
                                    result['viability_issue'] = True
                                else:
                                    result['viability_issue'] = False
                                    
                                results.append(result)
                
                # Create results dataframe
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.write("Pricing recommendations generated for all products:")
                    st.dataframe(
                        results_df.style.apply(
                            lambda x: ['background-color: #ffcccc' 
                                      if x['viability_issue'] else '' 
                                      for i in range(len(x))], 
                            axis=1
                        )
                    )
                    
                    # Save results
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    try:
                        # Save to JSON format
                        filepath = os.path.join('pricing_strategies', f"batch_pricing_{timestamp}.json")
                        os.makedirs('pricing_strategies', exist_ok=True)
                        
                        with open(filepath, 'w') as f:
                            json.dump(results, f, indent=4)
                        
                        st.success(f"Results saved to {filepath}")
                        
                        # Add download button
                        st.markdown(get_download_link(results_df, f"pricing_results_{timestamp}.csv", 
                                                   "Download Results as CSV"), 
                                   unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Unable to save results to file: {str(e)}")
                        # Still provide the download option
                        st.markdown(get_download_link(results_df, f"pricing_results_{timestamp}.csv", 
                                                   "Download Results as CSV"), 
                                   unsafe_allow_html=True)
                else:
                    st.error("No valid products to process. Please check your data and try again.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 3: Market Analysis
with tab3:
    st.header("Market Analysis")
    
    # Category selection
    analysis_category = st.selectbox("Select Category for Analysis", available_categories, key="analysis_category")
    
    # Get market statistics
    if analysis_category:
        # Get market statistics from category benchmarks
        market_stats = {}
        
        if analysis_category in st.session_state.pricing_strategy.category_benchmarks:
            benchmark = st.session_state.pricing_strategy.category_benchmarks[analysis_category]
            q1 = benchmark.get('q1', 0)
            q3 = benchmark.get('q3', 0)
            lower = benchmark.get('lower_bound', 0)
            upper = benchmark.get('upper_bound', 0)
            
            # Calculate additional statistics
            median = (q1 + q3) / 2
            iqr = q3 - q1
            min_price = max(0, lower)
            max_price = upper
            
            market_stats[analysis_category] = {
                'median_price': median,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'min_price': min_price,
                'max_price': max_price
            }
            
            # Create layout
            col1, col2 = st.columns(2)
            
            # Extract data for selected category
            cat_stats = market_stats[analysis_category]
            
            # Display summary statistics
            with col1:
                st.subheader("Price Distribution")
                
                # Create pretty metrics
                metric_cols = st.columns(2)
                with metric_cols[0]:
                    st.metric("Median Price", f"₹{cat_stats.get('median_price', 0):.2f}")
                    st.metric("25th Percentile (Q1)", f"₹{cat_stats.get('q1', 0):.2f}")
                    st.metric("75th Percentile (Q3)", f"₹{cat_stats.get('q3', 0):.2f}")
                
                with metric_cols[1]:
                    st.metric("Price Range", f"₹{cat_stats.get('max_price', 0) - cat_stats.get('min_price', 0):.2f}")
                    st.metric("Min Price", f"₹{cat_stats.get('min_price', 0):.2f}")
                    st.metric("Max Price", f"₹{cat_stats.get('max_price', 0):.2f}")
                
                # Show distribution visualization
                try:
                    img_path = f"visualizations/{analysis_category}_price_distribution.png"
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        # Create a simple visualization instead
                        fig, ax = plt.subplots(figsize=(8, 4))
                        x = np.linspace(cat_stats['min_price'], cat_stats['max_price'], 1000)
                        # Simulate a normal distribution using the IQR
                        std_dev = cat_stats['iqr'] / 1.35
                        mean = cat_stats['median_price']
                        y = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
                        ax.plot(x, y)
                        ax.axvline(x=cat_stats['q1'], color='r', linestyle='--')
                        ax.axvline(x=cat_stats['median_price'], color='g', linestyle='-')
                        ax.axvline(x=cat_stats['q3'], color='r', linestyle='--')
                        ax.set_title(f'{analysis_category} Price Distribution (Estimated)')
                        ax.set_xlabel('Price (₹)')
                        ax.set_ylabel('Density')
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error displaying distribution: {str(e)}")
            
            # Display pricing strategy information
            with col2:
                st.subheader("Pricing Strategy Recommendations")
                
                # Pricing tiers
                st.write("#### Recommended Pricing Tiers")
                tiers = {
                    "Aggressive Entry": cat_stats['q1'] * 0.9,
                    "Value Tier": cat_stats['q1'],
                    "Competitive Tier": cat_stats['median_price'] * 0.9,
                    "Market Average": cat_stats['median_price'],
                    "Premium Tier": cat_stats['q3']
                }
                
                tier_df = pd.DataFrame({
                    'Pricing Tier': list(tiers.keys()),
                    'Price Point': [f"₹{v:.2f}" for v in tiers.values()],
                    'Strategy': [
                        "Maximum customer acquisition",
                        "High volume, low margin",
                        "Balanced acquisition/margin",
                        "Standard market position",
                        "Premium positioning"
                    ]
                })
                
                st.table(tier_df)
                
                # Manufacturing cost viability
                st.write("#### Manufacturing Cost Viability")
                
                # Get category thresholds
                warning, viability = st.session_state.pricing_strategy.category_thresholds.get(
                    analysis_category, (0.85, 1.0))
                
                # Calculate threshold prices
                warning_price = cat_stats['median_price'] * warning
                viability_price = cat_stats['median_price'] * viability
                min_margin = st.session_state.pricing_strategy.category_min_margins.get(
                    analysis_category, st.session_state.pricing_strategy.min_profit_margin)
                
                st.write(f"""
                For successful entry in the {analysis_category} category:
                - Keep manufacturing costs below: ₹{warning_price:.2f}
                - Manufacturing costs above ₹{viability_price:.2f} may not be viable
                - Ensure at least {min_margin*100:.1f}% profit margin for sustainability
                """)
                
                # Create manufacturing cost slider
                cost_input = st.slider(
                    "Test Manufacturing Cost", 
                    min_value=float(cat_stats['min_price']/3), 
                    max_value=float(cat_stats['median_price']*1.5),
                    value=float(cat_stats['median_price']*0.6),
                    step=float(cat_stats['median_price']/20)
                )
                
                # Calculate viability metrics
                cost_to_median = cost_input / cat_stats['median_price']
                min_viable_price = cost_input / (1 - min_margin)
                
                # Show metrics
                cost_metrics = st.columns(3)
                with cost_metrics[0]:
                    st.metric(
                        "Cost/Median Ratio", 
                        f"{cost_to_median:.2f}x",
                        delta=None if 0.5 <= cost_to_median <= 0.7 else 
                              f"{cost_to_median - 0.6:.2f}",
                        delta_color="normal" if 0.5 <= cost_to_median <= 0.7 else "inverse"
                    )
                
                with cost_metrics[1]:
                    st.metric(
                        "Minimum Viable Price",
                        f"₹{min_viable_price:.2f}"
                    )
                
                with cost_metrics[2]:
                    viable_tier = "Aggressive" if min_viable_price < tiers["Value Tier"] else \
                                 "Value" if min_viable_price < tiers["Competitive Tier"] else \
                                 "Competitive" if min_viable_price < tiers["Market Average"] else \
                                 "Premium" if min_viable_price < tiers["Premium Tier"] else \
                                 "Not Viable"
                    st.metric("Viable Pricing Tier", viable_tier)
                
                # Show viability status
                if cost_to_median < warning:
                    st.success("✅ This manufacturing cost allows for competitive pricing")
                elif cost_to_median < viability:
                    st.warning("⚠️ This manufacturing cost limits pricing flexibility")
                else:
                    st.error("❌ This manufacturing cost may not be viable for market entry")
        else:
            st.warning(f"No market data available for {analysis_category}")

# Add a batch upload page for pricing strategies
with tab4:
    st.header("Pricing Strategy History")
    
    # Show previously saved pricing strategies
    try:
        pricing_files = []
        
        # Check if directory exists
        if os.path.exists('pricing_strategies'):
            pricing_files = sorted(
                [f for f in os.listdir('pricing_strategies') if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join('pricing_strategies', x)),
                reverse=True
            )
        
        if pricing_files:
            st.write(f"Found {len(pricing_files)} saved pricing strategies")
            
            # Create a filter by category
            all_saved_categories = set()
            
            # Read the first few strategies to extract categories
            for file in pricing_files[:min(len(pricing_files), 20)]:
                try:
                    with open(os.path.join('pricing_strategies', file), 'r') as f:
                        strategy_data = json.load(f)
                        all_saved_categories.add(strategy_data.get('category', 'Unknown'))
                except:
                    pass
            
            # Filter options
            filter_category = st.selectbox(
                "Filter by Category",
                options=["All Categories"] + sorted(list(all_saved_categories))
            )
            
            # Display strategies
            for file in pricing_files[:20]:  # Limit to 20 to avoid performance issues
                try:
                    file_path = os.path.join('pricing_strategies', file)
                    with open(file_path, 'r') as f:
                        strategy_data = json.load(f)
                    
                    # Apply filter
                    if filter_category != "All Categories" and strategy_data.get('category') != filter_category:
                        continue
                    
                    # Check if it's a batch or individual strategy
                    if isinstance(strategy_data, list):
                        # Batch result
                        st.subheader(f"Batch Pricing: {file}")
                        
                        # Convert to DataFrame and display
                        batch_df = pd.DataFrame(strategy_data)
                        st.dataframe(batch_df)
                        
                        # Add download link
                        st.markdown(get_download_link(batch_df, file.replace('.json', '.csv'), 
                                                   "Download as CSV"), 
                                   unsafe_allow_html=True)
                    else:
                        # Individual strategy
                        with st.expander(f"{strategy_data.get('category', 'Unknown')} - {file}"):
                            # Display key metrics
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Market Price", f"₹{strategy_data.get('predicted_market_price', 0):.2f}")
                                st.metric("Manufacturing Cost", f"₹{strategy_data.get('manufacturing_cost', 0):.2f}")
                            
                            with cols[1]:
                                st.metric("Recommended Price", f"₹{strategy_data.get('recommended_price', 0):.2f}")
                                st.metric("Discount from Market", f"{strategy_data.get('discount_from_market', 0):.1f}%")
                            
                            with cols[2]:
                                st.metric("Profit Margin", f"{strategy_data.get('profit_margin_percentage', 0):.1f}%")
                                st.metric("Strategy", strategy_data.get('strategy', 'Unknown'))
                            
                            # Add download button for individual strategy
                            st.json(strategy_data)
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
        else:
            st.info("No pricing strategies saved yet. Generate some recommendations to see them here.")
    except Exception as e:
        st.error(f"Error loading pricing strategies: {str(e)}")

# Tab 5: Category Settings
with tab5:
    st.header("Category-Specific Settings")
    
    st.info("""
    Each product category has custom settings optimized for its unique market characteristics.
    These settings affect pricing recommendations and help ensure they are appropriate for each category.
    """)
    
    if hasattr(st.session_state, 'pricing_strategy'):
        strategy = st.session_state.pricing_strategy
        
        # Create settings table
        settings_data = []
        
        for category in available_categories:
            # Get settings
            calibration = strategy.calibration_factors.get(category, 1.0)
            min_margin = strategy.category_min_margins.get(category, strategy.min_profit_margin) * 100
            warning, viability = strategy.category_thresholds.get(category, (0.85, 1.0))
            
            # Market benchmark
            category_benchmark = {}
            if hasattr(strategy, 'category_benchmarks'):
                category_benchmark = strategy.category_benchmarks.get(category, {})
            
            q1 = category_benchmark.get('q1', 0)
            q3 = category_benchmark.get('q3', 0)
            
            # Add to list
            settings_data.append({
                'Category': category,
                'Calibration Factor': f"{calibration:.2f}x",
                'Min Profit Margin': f"{min_margin:.1f}%",
                'Warning Threshold': f"{warning:.2f}x",
                'Viability Threshold': f"{viability:.2f}x",
                'Q1 Benchmark': f"₹{q1:.2f}",
                'Q3 Benchmark': f"₹{q3:.2f}"
            })
        
        # Convert to DataFrame and display
        settings_df = pd.DataFrame(settings_data)
        st.dataframe(settings_df, use_container_width=True)
        
        # Explanations
        with st.expander("What do these settings mean?"):
            st.markdown("""
            **Calibration Factor**: Adjusts price predictions to match real market prices for this category.
            Higher values mean our model typically underpredicts prices for this category.
            
            **Min Profit Margin**: The minimum acceptable profit margin for this category.
            Categories with higher brand value can sustain higher margins.
            
            **Warning Threshold**: When manufacturing cost exceeds this percentage of the median price,
            pricing flexibility becomes limited. Shows as a warning in recommendations.
            
            **Viability Threshold**: When manufacturing cost exceeds this percentage of the median price,
            the product may not be viable at all. Shows as an error in recommendations.
            
            **Q1/Q3 Benchmarks**: 25th and 75th percentile price points for this category.
            Used to position products in the market and evaluate competitiveness.
            """)
    else:
        st.error("Pricing strategy not initialized")

# Footer
st.markdown("---")
st.markdown("Competitive Pricing Strategy Tool - Built with Streamlit")
st.markdown("Version 2.0 - Enhanced with category-specific calibration, margins and thresholds") 