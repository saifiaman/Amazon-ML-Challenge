import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import random

# Simulated feature extraction from images (you can replace this with actual image processing)
def extract_image_features(image_link):
    # In a real scenario, you would download the image and extract features using a CNN
    return [random.random() for _ in range(10)]  # Dummy feature vector

# Function to prepare the training dataset
def prepare_data(train_df):
    X = []
    y = []
    
    for _, row in train_df.iterrows():
        image_features = extract_image_features(row['image_link'])
        X.append(image_features + [row['group_id'], row['entity_name']])
        y.append(row['entity_value'].split()[0])  # Extract the numerical value from the entity_value
    
    return X, y

# Model training
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Prediction function
def predictor(model, image_link, group_id, entity_name):
    image_features = extract_image_features(image_link)
    input_data = image_features + [group_id, entity_name]
    
    # Make a prediction using the trained model
    prediction = model.predict([input_data])[0]
    
    # Format the prediction (this part can be improved by handling unit prediction better)
    unit = "inch"  # Assuming we predict a unit like "inch" for now (improve this for real-world cases)
    return f"{prediction:.2f} {unit}"

# Main function to generate predictions for test data
def generate_predictions(model, test_df):
    test_df['prediction'] = test_df.apply(
        lambda row: predictor(model, row['image_link'], row['group_id'], row['entity_name']), axis=1)
    return test_df

if __name__ == "__main__":
    # Load the dataset
    DATASET_FOLDER = '../dataset/'
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Prepare training data
    X, y = prepare_data(train_df)
    
    # Train the model
    model = train_model(X, y)
    
    # Generate predictions for the test set
    test_df = generate_predictions(model, test_df)
    
    # Save the predictions to a CSV file
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test_df[['index', 'prediction']].to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")
