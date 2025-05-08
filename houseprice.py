# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Load data
data = pd.read_csv("C:\\Users\\boddu\\Downloads\\train.csv")  # Make sure 'train.csv' is in your project folder

# Step 3: Prepare data (use only square footage vs price)
X = data[["GrLivArea"]]  # Square footage
y = data["SalePrice"]    # Price

# Step 4: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict prices
predicted_prices = model.predict(X)

# Step 6: Plot results
plt.scatter(X, y, color="blue", label="Actual Prices")
plt.plot(X, predicted_prices, color="red", label="Predicted Prices")
plt.xlabel("Square Footage (sqft)")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# Step 7: Predict a new house (e.g., 1500 sqft)
new_house = [[1500]]
predicted_price = model.predict(new_house)
print(f"Predicted price for 1500 sqft: ${predicted_price[0]:,.2f}")