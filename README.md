```markdown
# Iris Flower Classification

## Overview
The **Iris Flower Classification** project focuses on classifying iris flowers into one of three species—**setosa**, **versicolor**, and **virginica**—based on their measurements. Using machine learning techniques, we train a model to predict the species of an iris flower from its features like sepal length, sepal width, petal length, and petal width.

---

## Dataset
The dataset includes 150 samples of iris flowers, with the following features:
- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**
- **Species (setosa, versicolor, virginica)**

### Source
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv).

---

## Tools and Libraries Used
- **Python 3**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computation
- **Seaborn**: Data visualization
- **Matplotlib**: Plotting graphs
- **Scikit-learn**: Machine learning library

---

## Project Workflow
1. **Data Loading and Exploration**:
   - Load the dataset using Pandas.
   - Perform exploratory data analysis (EDA) to understand the distribution and correlation of features.

2. **Visualization**:
   - Box plots for each feature grouped by species.
   - Pair plots and scatter plots to visualize relationships between features.
   - Heatmap to show feature correlations.

3. **Preprocessing**:
   - Drop unnecessary columns (e.g., `Id`).
   - Encode categorical variables using One-Hot Encoding.

4. **Model Training**:
   - Split the data into training and testing sets (80:20).
   - Train a **Logistic Regression** model using the training set.

5. **Evaluation**:
   - Measure the model's performance using metrics like:
     - **Accuracy**: 1.0
     - **Precision**: 1.0
   - Generate confusion matrices and classification reports.

---

## Key Results
- **Accuracy**: The logistic regression model achieved an accuracy of **100%** on the test set.
- **Precision**: The weighted precision score is **1.0**, indicating perfect classification.

---

## Visualizations
### Box Plots
- Distribution of sepal and petal dimensions for each species.

### Scatter Plots
- Relationships between sepal length and width across species.

### Correlation Heatmap
- Visualized correlation between numerical features.

---

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone <repository-link>
   cd iris-flower-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open the `Iris_Flower_Classification.ipynb` notebook in Jupyter or Google Colab.

---

## References
- Dataset: [Kaggle Iris Dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- Documentation: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## Author
Developed by **D. Muni Tejo Venkata Sai**.
