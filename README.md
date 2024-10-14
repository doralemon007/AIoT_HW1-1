
# Introduction to Linear Regression: A CRISP-DM Guided Project

This project, implemented in `HW1_v2.py`, serves as an **educational tool** to introduce users to the **fundamentals of linear regression**. It follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to teach key concepts in data science while using **linear regression** as the core algorithm. The project is designed to be interactive and user-friendly, allowing learners to engage with core aspects of linear regression such as data preprocessing, model fitting, and evaluation.

## Key Objectives:
1. **Understand Linear Regression**: Explore how linear regression works and the impact of key variables such as slope, intercept, and noise.
2. **Follow the CRISP-DM Methodology**: Apply the structured steps of the CRISP-DM framework to ensure the logical flow from understanding the business problem to model deployment.
3. **Natural Language Programming**: Demonstrate how programming concepts and project goals can be built interactively through **natural language prompts**.

## Table of Contents
- [Overview](#overview)
- [CRISP-DM Methodology Applied](#crisp-dm-methodology-applied)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Running the Code](#running-the-code)
- [Author](#author)
- [Natural Language Programming with ChatGPT](#natural-language-programming-with-chatgpt)
- [Project Structure](#project-structure)

## Overview

This project offers an interactive and beginner-friendly approach to learning **linear regression** using the Python programming language. The project is designed for learners who are new to machine learning and want to gain an intuitive understanding of linear regression's core concepts. 

The project builds on the **CRISP-DM framework** to guide users through each stage of a data mining process:
- **Business Understanding**: Introduce linear regression as a tool to model the relationship between independent and dependent variables.
- **Data Understanding**: Generate synthetic data with noise to show how data distribution impacts model fitting.
- **Data Preparation**: Split data into training and testing sets, allowing users to explore how different splits affect model performance.
- **Modeling**: Implement linear regression using the `scikit-learn` library.
- **Evaluation**: Provide key metrics like **R-squared (R²)**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)** for both training and test data.
- **Deployment**: Visualize the model's predictions and compare them to the actual data.

## CRISP-DM Methodology Applied

This project closely follows the **CRISP-DM** process to ensure a structured approach to linear regression:

1. **Business Understanding**: 
   - Goal: Help users learn the basic principles of linear regression, emphasizing how noise impacts model performance.
   - Focus: Teach fundamental concepts like slope, intercept, and model evaluation.

2. **Data Understanding**:
   - Synthetic data is generated based on the equation: `y = aX + b + noise`, where noise follows a normal distribution.
   - The noise introduces variability, allowing users to see how the linear regression model behaves with different levels of noise.

3. **Data Preparation**:
   - Users can control the training-to-testing data split, adjusting the training ratio dynamically through sliders in the app.
   - The project also visualizes the data distributions (e.g., histograms of `x` and `y`) to help users understand the data characteristics before modeling.

4. **Modeling**:
   - The `LinearRegression` model from the **scikit-learn** library is used to fit the model to training data.
   - The project explains key modeling concepts and shows how to fit a line to the data using gradient descent or closed-form solutions.

5. **Evaluation**:
   - The model is evaluated using **R²**, **MSE**, and **MAE** on both training and test datasets.
   - Users can compare model performance across training and testing sets, understanding the impact of overfitting or underfitting.

6. **Deployment**:
   - The model’s predictions are plotted alongside the actual data.
   - A residual plot helps users visualize the difference between predicted and actual values.
   - Dynamic elements like **tooltips** and **tutorial mode** guide users through each step of the process.

## Key Features

### Natural Language Programming Approach:
This project was developed interactively using **natural language programming**. By specifying goals and outcomes in plain language, the programming process becomes collaborative and adaptive. The core elements, such as data generation, model building, and evaluation, were iteratively refined based on user feedback and project objectives.

### Interactive Learning:
- **Sliders** for controlling slope, intercept, noise, and the training-test split provide a hands-on learning experience.
- **Dynamic Visualizations**: See how changes to the data impact the regression line, model performance, and evaluation metrics.
- **Residual Plots**: Explore how well the model fits the data through residual analysis.

### Model Performance Metrics:
- **R-squared (R²)**: Measures how well the model explains the variance in the data.
- **Mean Squared Error (MSE)**: Shows the average squared difference between actual and predicted values.
- **Mean Absolute Error (MAE)**: Provides a simpler interpretation of the average error in prediction.

## Requirements

To run this project, you will need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `streamlit`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Code

1. **Clone the repository**:
   ```bash
   git clone [your repository URL]
   cd [your repository directory]
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run HW1_v2.py
   ```

3. **Interactive Mode**:
   - Once the app is running, use the sidebar sliders to control the model’s parameters (slope, intercept, noise, etc.).
   - Toggle **Tutorial Mode** for a guided step-by-step walkthrough.

4. **Visual Outputs**:
   - The app will generate **scatter plots**, **regression lines**, and **residual plots** to help users understand model performance.
   - Metrics for training and testing data will be displayed side by side.

## Project Structure

```plaintext
├── HW1_v2.py            # Main Python file for the project
├── README.md            # Project documentation (this file)
└── requirements.txt     # List of dependencies
```

## Author

- **Chien-Ming Chen**
- **Course**: NCHU CS AIoT

---

## Natural Language Programming with ChatGPT

This project was created through interactive prompts using **ChatGPT**. Below is a breakdown of how natural language programming was used to iteratively improve the project and achieve the final goal:

### Example of an Initial Prompt:
```
I'd like help fulfilling a homework project about machine learning and demo with Streamlit. Please follow my instructions step by step.
```
**Result**: ChatGPT helped initiate the project by setting up the required libraries, generating synthetic data, and explaining key concepts in linear regression.

### Example of a Refinement Prompt:
```
Write Python code to demonstrate how to solve a linear regression problem, following the CRISP data mining 6 steps, using the above function to get data points.
```
**Result**: ChatGPT structured the project around the **CRISP-DM methodology**, providing explanations and code for each stage, from data understanding to deployment.

### Rephrased Prompt for Feedback and Fine-Tuning:
```
Deploy the project using an interactive web app with Streamlit. Introduce linear regression to the user, allowing them to set variables and understand how noise impacts the result.
```
**Result**: ChatGPT generated a fully functional Streamlit app where users can interact with sliders to explore how noise, slope, and intercept impact the regression model.

### Example of a Code Refinement Request:
```
I'd like to add a residual plot to the project and make the tooltips dynamic, explaining R², MSE, and MAE metrics. Also, adjust label positioning dynamically based on the slope.
```
**Result**: ChatGPT added a residual plot, dynamically adjusted the label positioning based on the slope value, and provided tooltip explanations for model evaluation metrics.

---

By using **natural language programming**, this project evolved into a robust, interactive learning tool for those interested in linear regression and data science principles.
