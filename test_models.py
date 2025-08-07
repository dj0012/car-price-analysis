import joblib
import pandas as pd
import numpy as np

def test_prediction():
    """Test the trained models with sample data"""
    
    # Load models and data
    models = joblib.load('car_price_models.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    sample_data = joblib.load('sample_data.pkl')
    
    print("✅ Models loaded successfully!")
    print(f"Available models: {list(models.keys())}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Test with sample car data
    test_cases = [
        {
            'brand': 'Maruti',
            'year': 2020,
            'fuel_type': 'Petrol',
            'transmission_type': 'Manual',
            'seller_type': 'Individual',
            'km_driven': 30000,
            'engine': 1197,
            'max_power': 82.0,
            'mileage': 18.9,
            'seats': 5
        },
        {
            'brand': 'Hyundai',
            'year': 2019,
            'fuel_type': 'Diesel',
            'transmission_type': 'Manual',
            'seller_type': 'Dealer',
            'km_driven': 45000,
            'engine': 1582,
            'max_power': 126.0,
            'mileage': 22.3,
            'seats': 5
        },
        {
            'brand': 'BMW',
            'year': 2018,
            'fuel_type': 'Petrol',
            'transmission_type': 'Automatic',
            'seller_type': 'Dealer',
            'km_driven': 25000,
            'engine': 1998,
            'max_power': 184.0,
            'mileage': 14.8,
            'seats': 5
        }
    ]
    
    print("\n🔍 Testing predictions:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['brand']} {test_case['year']} ({test_case['fuel_type']})")
        
        # Create input features
        vehicle_age = 2025 - test_case['year']
        price_per_km = 1.0
        power_to_engine_ratio = test_case['max_power'] / test_case['engine']
        
        # Create base features
        input_data = {
            'vehicle_age': vehicle_age,
            'km_driven': test_case['km_driven'],
            'mileage': test_case['mileage'],
            'engine': test_case['engine'],
            'max_power': test_case['max_power'],
            'seats': test_case['seats'],
            'year': test_case['year'],
            'price_per_km': price_per_km,
            'power_to_engine_ratio': power_to_engine_ratio
        }
        
        # Add all categorical features with default 0
        for col in feature_columns:
            if col not in input_data:
                input_data[col] = 0
        
        # Set the selected categorical features to 1
        brand_col = f"brand_{test_case['brand']}"
        fuel_col = f"fuel_type_{test_case['fuel_type']}"
        transmission_col = f"transmission_type_{test_case['transmission_type']}"
        seller_col = f"seller_type_{test_case['seller_type']}"
        
        if brand_col in input_data:
            input_data[brand_col] = 1
        if fuel_col in input_data:
            input_data[fuel_col] = 1
        if transmission_col in input_data:
            input_data[transmission_col] = 1
        if seller_col in input_data:
            input_data[seller_col] = 1
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Make predictions
        xgb_pred = models['xgboost'].predict(input_df)[0]
        
        input_scaled = models['scaler'].transform(input_df)
        ridge_pred = models['ridge'].predict(input_scaled)[0]
        
        print(f"  XGBoost Prediction: ₹{xgb_pred:,.0f}")
        print(f"  Ridge Prediction: ₹{ridge_pred:,.0f}")
        print(f"  Difference: ₹{abs(xgb_pred - ridge_pred):,.0f}")
    
    print("\n✅ All tests completed successfully!")
    print("\nYou can now run the Streamlit app with:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    test_prediction()
