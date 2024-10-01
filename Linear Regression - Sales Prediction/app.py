from flask import Flask, request, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Load the data from CSV file
df = pd.read_csv('advertising_sales.csv')

# Prepare the data for regression
X = df[['Advertising_Spend']]
y = df['Sales']

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    advertising_spend = float(request.form['advertising_spend'])
    predicted_sales = model.predict([[advertising_spend]])[0]

    # Round the predicted sales to 2 decimal places
    predicted_sales = round(predicted_sales, 2)

    # Plotting
    plt.figure(figsize=(8, 5))  # Adjusted size for the plot
    plt.scatter(df['Advertising_Spend'], df['Sales'], color='blue', label='Actual Sales', s=100)
    plt.plot(df['Advertising_Spend'], model.predict(X), color='red', label='Predicted Sales', linewidth=2)
    plt.scatter(advertising_spend, predicted_sales, color='green', label='Input Point', s=200, edgecolor='black')
    plt.title('Advertising Spend vs Sales', fontsize=14, fontfamily='sans-serif', fontweight='bold')
    plt.xlabel('Advertising Spend (₹)', fontsize=12, fontfamily='sans-serif')
    plt.ylabel('Sales (₹)', fontsize=12, fontfamily='sans-serif')
    plt.legend(fontsize=10)
    plt.grid(True)

    # Save the plot
    plot_path = 'static/plot.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=100)  # Save with tight bounding box
    plt.close()

    return render_template('index.html', predicted_sales=predicted_sales, plot_path=plot_path)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
