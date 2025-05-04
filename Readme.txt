# Crop Yield Prediction App

## Overview

The Crop Yield Prediction App is a Streamlit-based web application designed to predict crop yields using machine learning models. It allows users to upload a dataset, preprocess the data, train various machine learning models, and visualize the predictions through interactive graphs. The app supports K-Nearest Neighbors (KNN), Random Forest, and Linear Regression models.

## Features

- Upload and preview CSV datasets.
- Preprocess data by handling missing values and encoding categorical variables.
- Train and evaluate machine learning models (KNN, Random Forest, Linear Regression).
- Visualize predictions with interactive graphs.
- Display model evaluation metrics (Mean Squared Error, R-squared).
- Compare the performance of different models.
- Download trained models and predictions as CSV files.

## Setup Instructions

### Prerequisites

- Python 3.7 or later
- Pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction-app.git
   cd crop-yield-prediction-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the app** in your web browser at `http://localhost:8501`.

## Usage

1. **Upload a Dataset**: Use the file uploader to load a CSV dataset containing crop yield data.
2. **Preprocess Data**: The app automatically handles missing values and encodes categorical variables.
3. **Select a Model**: Choose from KNN, Random Forest, or Linear Regression.
4. **Train the Model**: Click the "Train the model" button to train the selected model on your dataset.
5. **View Results**: The app displays evaluation metrics and interactive graphs to visualize the predictions.
6. **Compare Models**: The app compares the performance of different models and displays their R-squared scores.
7. **Download Results**: You can download the trained model and predictions as CSV files.

## Project Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: A list of Python packages required to run the application.
- `README.md`: This file, providing an overview of the project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.


## Contact

For any questions or feedback, please contact (mailto:t.shaiknaushad@gmail.com).