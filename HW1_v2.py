import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Business Understanding
st.title("Linear Regression Demonstration")
st.write("""
## Step 1: Business Understanding
This app demonstrates how linear regression works and how noise impacts the model's performance. 
You will see how adding noise to the data affects the ability of the model to accurately predict the relationship between `x` and `y`.
""")

# Step-by-step tutorial mode
st.sidebar.title("Step-by-Step Tutorial")
tutorial = st.sidebar.checkbox("Activate Step-by-Step Tutorial")

if tutorial:
    st.sidebar.write("""
    ### Tutorial Guide:
    1. **Set Parameters**: Use the sliders to adjust the slope (`a`), intercept (`b`), noise, and train/test ratio.
    2. **Visualize Data**: Look at the histograms and a random sample of your data.
    3. **Train the Model**: Fit a linear regression model to your data.
    4. **Evaluate the Model**: Analyze the performance of the model with metrics like R², MSE, and MAE.
    5. **Residual Analysis**: Look at the residual plot to see how well the model fits the data.
    """)

# Sidebar for user inputs (Step 3: Data Preparation)
st.sidebar.title("Set Parameters")

n = st.sidebar.slider("Number of points (n)", 10, 500, 100)
a = st.sidebar.slider("Slope (a)", -10.0, 10.0, 1.0, step=0.1)
b = st.sidebar.slider("Intercept (b)", 1.0, 50.0, 5.0)
c = st.sidebar.slider("Noise Multiplier (c)", 0.0, 10.0, 5.0, step=0.2)
mean_noise = st.sidebar.slider("Mean of noise", 0.0, 10.0, 5.0, step=0.2)
variance_noise = st.sidebar.slider("Variance of noise", 1.0, 25.0, 5.0, step=0.5)
train_ratio = st.sidebar.slider("Training Data Ratio", 0.5, 0.9, 0.8)

# Data Generation Function
def generate_data(a, b, c, mean, variance, num_points=100, x_range=(0, 10)):
    x = np.linspace(x_range[0], x_range[1], num_points)
    noise = np.random.normal(mean, np.sqrt(variance), num_points)
    y = a * x + b + c * noise
    return x, y

# Generate the data based on user inputs
x, y = generate_data(a, b, c, mean_noise, variance_noise, n)

# Convert x and y into a DataFrame for easy statistics
data = pd.DataFrame({"x": x, "y": y})

# Display statistics about x and y
st.write("### Data Statistics")
st.write(data.describe())

# Plot histograms of x and y
st.write("### Histogram of X and Y")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(x, bins=20, color='blue', alpha=0.7)
ax1.set_title("Histogram of X")
ax1.set_xlabel("X")
ax1.set_ylabel("Frequency")

ax2.hist(y, bins=20, color='green', alpha=0.7)
ax2.set_title("Histogram of Y")
ax2.set_xlabel("Y")
ax2.set_ylabel("Frequency")
st.pyplot(fig)

# Show a random sample of the data and its table
st.write("### Random Sample of Data Points")

# Randomly sample 10 data points
sampled_data = data.sample(10).sort_index()

# Plot the sampled data (smaller scatter plot)
col1, col2 = st.columns(2)
with col1:
    st.write("#### Scatter Plot of Sampled Points")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(sampled_data["x"], sampled_data["y"], color='purple')
    ax.set_title("Sampled Data Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)

# Display the sampled data as a table in the adjacent column
with col2:
    st.write("#### Sampled Data Points Table")
    st.dataframe(sampled_data.reset_index(drop=True))

# Step 3: Data Preparation
st.write("## Step 3: Data Preparation")
X = x.reshape(-1, 1)

# Split the data into training and testing sets based on the user-defined ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, test_size=1-train_ratio, random_state=42)

st.write(f"Training data size: {X_train.shape[0]}, Testing data size: {X_test.shape[0]}")

# Step 4: Modeling
st.write("## Step 4: Modeling")

st.write("""
Linear regression is a simple algorithm that models the relationship between a dependent variable `y` and an independent variable `x`.
In this demonstration, the **LinearRegression** model from the `scikit-learn` library is used to fit a line through the data by minimizing the error (difference between predicted and actual values).
""")

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions for training data
y_train_pred = linear_model.predict(X_train)

# Show the coefficients from the trained model on training data
training_mse = mean_squared_error(y_train, y_train_pred)
training_r2 = r2_score(y_train, y_train_pred)
training_mae = mean_absolute_error(y_train, y_train_pred)

st.write(f"### Training Data Metrics:")
st.write(f" - **Coefficient (a)**: {linear_model.coef_[0]:.4f}")
st.write(f" - **Intercept (b)**: {linear_model.intercept_:.4f}")
st.write(f" - **R-squared (R²)**: {training_r2:.4f}")
st.write(f" - **Mean Squared Error (MSE)**: {training_mse:.4f}")
st.write(f" - **Mean Absolute Error (MAE)**: {training_mae:.4f}")

# Add sidebar tooltips for metrics explanation
if tutorial:
    st.sidebar.title("Metrics Explanation")
    st.sidebar.info("""
    **R²**: Proportion of variance in `y` explained by `x`. Higher values indicate a better fit.
    **MSE**: Average of the squared differences between actual and predicted values. Lower is better.
    **MAE**: Average of the absolute differences between actual and predicted values. Lower is better.
    """)

# Step 5: Evaluation
st.write("## Step 5: Evaluation")

# Predict on the test data
y_test_pred = linear_model.predict(X_test)

# Calculate evaluation metrics for the test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Display training and test metrics side by side
st.write("### Comparison of Training and Test Metrics")
comparison_data = {
    "Metric": ["R²", "MSE", "MAE"],
    "Training Data": [training_r2, training_mse, training_mae],
    "Test Data": [test_r2, test_mse, test_mae]
}
comparison_df = pd.DataFrame(comparison_data)
st.table(comparison_df)

# Residual plot
st.write("### Residual Plot")
residuals = y_test - y_test_pred
fig, ax = plt.subplots(figsize=(8, 2))  # Shrink height by 50%
ax.scatter(X_test, residuals, color="blue")
ax.axhline(y=0, color='red', linestyle='--')
ax.set_title("Residuals (Test Data)")
ax.set_xlabel("X")
ax.set_ylabel("Residuals (Actual - Predicted)")
st.pyplot(fig)

st.write("Residuals show the difference between actual and predicted values. Ideally, residuals should be a=randomly distributed around 0, indicating a good model fit.")

# Step 6: Deployment
st.write("## Step 6: Deployment")
st.write("You can see the generated data points and the fitted regression line below.")

# Plotting with dynamic label and metric positioning
fig, ax = plt.subplots()

# Plot the test data and predicted regression line
ax.scatter(X_test, y_test, color="blue", label="Test Data")
ax.plot(X_test, y_test_pred, color="red", label="Regression Line (model)")

# Add the actual line with slope `a` and intercept `b` (purple dashed line)

actual_line = a * X_test[::5] + b
ax.plot(X_test[::5], actual_line, color="purple", linestyle="--", label=f"Actual Line: \n(actual a={a}, actual b={b})")

# Dynamic label and metric positioning based on slope a
metrics_text = (
    f"R² (Test) = {test_r2:.4f}\n"
    f"MSE (Test) = {test_mse:.4f}\n"
    f"MAE (Test) = {test_mae:.4f}\n"
    f"Slope (a): {linear_model.coef_[0]:.4f}\n"
    f"Intercept (b): {linear_model.intercept_:.4f}"
    #f"Slope (a): {linear_model.coef_[0]:.4f} (Actual a: {a:.2f})\n"
    #f"Intercept (b): {linear_model.intercept_:.4f} (Actual b: {b:.2f})"
)

if a > 0:
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))
    ax.legend(loc="lower right")
else:
    ax.text(0.05, 0.05, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle="round", alpha=0.1))
    ax.legend(loc="upper right")

# Set title and labels
ax.set_title("Linear Regression on Test Data")
ax.set_xlabel("X")
ax.set_ylabel("y")

# Show plot
st.pyplot(fig)

st.write("This is a simple demonstration of the linear regression process using the CRISP-DM framework.")
