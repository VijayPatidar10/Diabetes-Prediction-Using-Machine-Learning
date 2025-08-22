# Diabetes-Prediction-Using-Machine-Learning
## 🛠️ Methodology

### 1. Data Cleaning
The initial exploration revealed that several columns contained zero values, which are not biologically plausible for measurements like Glucose, Blood Pressure, or BMI. These zeros were treated as missing data and were replaced with the **mean** of their respective columns to maintain data integrity.

### 2. Data Visualization
To better understand the data, the following visualizations were created:
* A **pie chart** and a **countplot** to show the distribution of the `Outcome` variable.
* **Histograms** for each feature to visualize their distributions.
* **Scatter plots** to explore the relationships between different features.

### 3. Model Training & Prediction
A **Decision Tree Classifier** was chosen as the predictive model for this task. The workflow was as follows:
* The data was split into features (`X`) and the target variable (`y`).
* The dataset was then divided into a training set and a testing set.
* The Decision Tree model was trained on the training data.
* The model's accuracy was evaluated, and it was then used to make predictions on new input data.

## 🚀 How to Run This Project

To run this project on your local machine, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/VijayPatidar10/Diabetes-Prediction-Using-Machine-Learning.git
    ```
2.  **Navigate to the project directory**:
    ```bash
    cd Diabetes-Prediction-Using-Machine-Learning
    ```
3.  **Install the required libraries**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook DiabetesPredictionUsingMachineLearning.ipynb
    ```
5.  Run the cells in the notebook to see the analysis and predictions.

## 📈 Results

The trained model can take a new set of health measurements as input and predict whether the individual is likely to be diabetic or not. The notebook includes an example of how to format new data for prediction.
