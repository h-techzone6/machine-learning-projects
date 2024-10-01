from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from CSV
df = pd.read_csv('housing_data.csv')

# Train the model
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        square_feet = float(request.form['squareFeet'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        
        # Make prediction
        input_data = pd.DataFrame({
            'SquareFeet': [square_feet],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms]
        })
        prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
