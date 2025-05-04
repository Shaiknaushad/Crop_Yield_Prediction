


# 🌾 Crop Yield Prediction

A machine learning web application that predicts crop yield based on various environmental and agricultural features such as rainfall, temperature, soil type, and more.

## 🚀 Features

- Upload CSV data for training
- Choose between multiple ML models: KNN, Random Forest, Linear Regression
- Hyperparameter tuning with GridSearchCV
- Visualize predictions with Matplotlib
- Download prediction results as CSV
- Clean UI powered by Streamlit

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

## 📂 Project Structure

```

Crop\_Yield\_Prediction/
├── app.py                  # Main Streamlit app
├── utils.py                # Utility functions
├── models.py               # Model training and prediction
├── data/                   # Sample datasets
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

````

## 📈 How It Works

1. Upload a CSV file with agricultural data.
2. Choose a model and configure parameters (if needed).
3. Train the model and visualize predictions.
4. Evaluate performance using MSE and R² score.

## ✅ Sample Input Format

Your CSV should look like this:

| temperature | rainfall | soil_type | fertilizer | yield |
|-------------|----------|-----------|------------|--------|
| 27.4        | 180      | Clay      | High       | 2.5    |

Make sure the `yield` column is included for training.

## 🧪 Running the App Locally

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/Crop_Yield_Prediction.git
cd Crop_Yield_Prediction
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run app.py
```

## 📦 Deployment

You can deploy this app using platforms like:

* [Streamlit Community Cloud](https://streamlit.io/cloud)
* [Heroku](https://www.heroku.com/)
* [Render](https://render.com/)

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📜 License

This project is open source under the MIT License.

---

**Made with ❤️ for smarter farming**

