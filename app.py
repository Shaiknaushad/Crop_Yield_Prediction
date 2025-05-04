import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to load a CSV file uploaded by the user
def load_uploaded_data():
    st.title('Crop Yield Prediction App')

    uploaded_file = st.file_uploader("Upload the Dataset", type="csv")

    if uploaded_file is not None:
        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Show a preview of the dataset
        st.write("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        return df
    else:
        st.warning("Please upload a CSV file to continue.")
        return None

# Function to preprocess the data
def preprocess_data(df, target_col="yield"):
    # Handling missing values and encoding categorical data
    df = df.dropna()  # Basic approach: drop rows with missing values
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

# Function to train and predict with the selected model
def train_and_predict(df, target_col="yield", model_type="KNN", n_neighbors=5):
    # Separate features and target variable
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model based on user selection
    if model_type == "KNN":
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "Linear Regression":
        model = LinearRegression()

    # Hyperparameter tuning for KNN
    if model_type == "KNN":
        param_grid = {'n_neighbors': [n_neighbors]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_test, y_pred, mse, r2, model

# Function to plot interactive graphs using Plotly
def plot_interactive_graphs(y_test, y_pred, df):
    # Actual vs Predicted Plot using Plotly
    st.subheader('Interactive Predicted vs Actual Yield')
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Yield', 'y': 'Predicted Yield'},
                     title="Predicted vs Actual Yield Comparison")
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                  line=dict(color="red", width=2))
    st.plotly_chart(fig)

    # Residual Plot
    residuals = y_test - y_pred
    st.subheader('Residuals')
    fig = px.histogram(residuals, nbins=30, labels={'value': 'Residuals'}, title="Residuals Distribution")
    st.plotly_chart(fig)

    # Check if 'crop_type' exists before plotting
    if "crop_type" in df.columns:
        # Crop Yield Distribution by Crop Type (Pie chart)
        st.subheader('Crop Yield Distribution')
        crop_yield_dist = df.groupby("crop_type")["yield"].mean().reset_index()
        fig = px.pie(crop_yield_dist, names='crop_type', values='yield', title="Crop Yield Distribution by Type")
        st.plotly_chart(fig)
    else:
        st.warning("Column 'crop_type' not found in the dataset.")

    # State-Wise Crop Yield
    st.subheader('State-Wise Crop Yield')
    state_yield = df.groupby("state")["yield"].mean().reset_index()
    fig = px.bar(state_yield, x='state', y='yield', title="Average Crop Yield by State")
    st.plotly_chart(fig)

    # Heatmap of crop yield distribution across regions
    st.subheader('State-Wise Crop Yield Heatmap')

    # Corrected pivot function call
    try:
        yield_pivot = df.pivot(index="state", columns="crop_type", values="yield")
        fig = px.imshow(yield_pivot, title="Heatmap of Crop Yield Across States and Crops")
        st.plotly_chart(fig)
    except KeyError as e:
        st.warning(f"Could not generate heatmap. Missing column: {e}")

# Function to plot cross-validation results
def plot_cross_val_results(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    st.subheader("Cross-Validation Results")
    st.write(f"Mean Cross-Validation Score: {scores.mean():.4f}")
    st.write(f"Standard Deviation: {scores.std():.4f}")

# Function to save the model
def save_model(model):
    filename = 'trained_model.joblib'
    joblib.dump(model, filename)
    st.download_button(label="Download Trained Model", data=open(filename, 'rb'), file_name=filename)

# Function to compare models
def compare_models(models, X, y):
    model_scores = {}

    # Loop through models and find the best one based on R-squared score
    for model_name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        model_scores[model_name] = r2

    # Find the best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = models[best_model_name]
    best_score = model_scores[best_model_name]

    return best_model_name, best_model, best_score, model_scores

# Main streamlit app
def main():
    # Load the dataset uploaded by the user
    df = load_uploaded_data()

    if df is not None:
        # Preprocess the data
        df = preprocess_data(df)

        # Model selection dropdown
        model_type = st.selectbox("Select Model", ["KNN", "Random Forest", "Linear Regression"])

        # User can adjust the number of neighbors for KNN
        if model_type == "KNN":
            n_neighbors = st.slider("Select Number of Neighbors for KNN", 1, 20, 5)
        else:
            n_neighbors = 5  # default for other models

        # Display a loading spinner while the model trains
        with st.spinner("Training the model... Please wait."):
            # Train the model and get predictions
            y_test, y_pred, mse, r2, trained_model = train_and_predict(df, target_col="yield", model_type=model_type, n_neighbors=n_neighbors)

        # Show model evaluation metrics
        st.write("### Model Evaluation Metrics")
        st.write(f"**Mean Squared Error (MSE):** {mse}")
        st.write(f"**R-squared (R2):** {r2}")

        # Plot interactive graphs
        plot_interactive_graphs(y_test, y_pred, df)

        # Show cross-validation results
        plot_cross_val_results(trained_model, df.drop(columns=["yield"]), df["yield"])

        # Allow user to download the trained model
        save_model(trained_model)

        # Allow user to download the predictions as CSV
        result_df = pd.DataFrame({"True Yield": y_test, "Predicted Yield": y_pred})
        csv = result_df.to_csv(index=False)
        st.download_button(label="Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        # Example usage with multiple models
        models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor()
        }

        # Now that df is available, call compare_models inside main()
        best_model_name, best_model, best_score, model_scores = compare_models(models, df.drop(columns=["yield"]), df["yield"])
        st.write(f"Best Model: {best_model_name} with R2 Score: {best_score:.4f}")

        # Display accuracy for each model
        st.write("### Model Accuracy (R-squared)")
        for model_name, score in model_scores.items():
            st.write(f"**{model_name}:** {score:.4f}")

if __name__ == "__main__":
    main()
