📌 Overview

This project implements a Long Short-Term Memory (LSTM) model to forecast energy consumption using historical time series data. The model captures temporal dependencies and sequential patterns, resulting in more accurate predictions compared to traditional statistical methods.

🎯 Objectives
Develop a deep learning model for time series forecasting
Analyze patterns in energy consumption data
Improve prediction accuracy using LSTM networks
Compare performance with traditional models
🧠 Model Description

The model is based on Recurrent Neural Networks (RNN), specifically LSTM architecture.

LSTM layers capture long-term dependencies
Uses tanh and sigmoid activation functions
Dense layer produces final output
LSTM Gates:
Forget Gate: Removes irrelevant information
Input Gate: Selects useful information to store
Output Gate: Generates prediction
⚙️ Tech Stack
Language: Python 🐍
Libraries: TensorFlow, Keras, NumPy, Pandas
Visualization: Matplotlib 📊
Environment: Jupyter Notebook / VS Code
🔄 Methodology
Data Preprocessing
Handle missing values
Normalize dataset
Convert into time series sequences
Model Building
Add LSTM layers
Add Dense output layer
Training
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
Evaluation
Compare predicted vs actual values
Analyze using error metrics (RMSE, MAE)
📈 Results
Achieves accurate forecasting of energy consumption
Effectively captures temporal patterns
Performs better than traditional regression models
🚀 Installation & Usage
# Clone the repository
git clone https://github.com/tanmoytalukdar98/Energy-Consumption-Forecasting-using-LSTM.git

# Navigate to the project directory
cd Energy-Consumption-Forecasting-using-LSTM

# Install required dependencies
pip install -r requirements.txt

# Run the project
python main.py
🔮 Future Scope
Implement Bi-LSTM for improved learning
Perform hyperparameter tuning
Build a real-time prediction system
Integrate with smart grid / IoT systems
🤝 Contribution

Contributions are welcome. Fork the repository and submit a pull request for improvements.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Tanmoy Talukdar
Gautam Kumar
Arnav Srivastava
Harsh Bhardwaj
