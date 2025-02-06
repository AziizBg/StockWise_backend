import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Function to simulate order dates
def datesO(n, lambd):
    T = np.random.poisson(lambd, n)
    D = np.cumsum(T)
    return D

# Function to determine the number of orders executed before a given date d
def beforeD(d, D):
    n = len(D)
    k = 0
    for i in range(n):
        if D[i] <= d:
            k += 1
        else:
            break
    return k

# Function to simulate ordered quantities or values (i.e., price)
# returns a DataFrame with columns Date, Quantity, and C_Quantity
def quantityO(n, lambd, d, m, sigma):
    D = datesO(n, lambd) # Simulate order dates 
    k = beforeD(d, D) # Number of orders executed before date d: due date
    D = D[:k] # Keep only the dates before date d
    Q = np.random.binomial(n * lambd, 1 - sigma**2 / m, k) # Simulate quantities ordered by the function binomial 
    Q = np.cumsum(Q) # Cumulative sum of the quantities ordered
    Q = np.round(Q) # Round the quantities to the nearest integer
    # np.insert inserts the initial quantity at the beginning of the array
    # np.diff calculates the difference between consecutive elements of the array
    out = pd.DataFrame({'Date': D, 'Quantity': np.insert(np.diff(Q), 0, Q[0]), 'C_Quantity': Q})
    return out

def SupplyA(delay, alpha, n, lambd, d, m, sigma):
    sales = quantityO(n, lambd, d, m, sigma)
    S1 = sales['Date'] - delay #S1 is the date of the supply which is the date of the order minus the delay
    S2 = np.round(sales['Quantity'] * (1 + alpha)) #S2 is the quantity of the supply which is the quantity of the order plus the stock rate
    S3 = np.cumsum(S2) #S3 is the cumulative supply quantity
    # SC_Quantity is the cumulative supply quantity which is the cumulative sum of the quantity of the supply
    # QuantityOC is the quantity ordered by the customer which is the quantity of the order in the sales DataFrame 
    outsupply = pd.DataFrame({'S_Dates': S1, 'SC_Quantity': S3, 'Date': sales['Date'], 'QuantityOC': sales['C_Quantity']})
    return outsupply

# StockPriceReview(unit_purchase_price, unit_sale_price, benefit_rate, expences_F, cumulative_supply, cumulative_quantity_ordered,quantity_ordered, dates)
def StockPriceReview(start_date, u_price, u_sale, benefit_rate, expences_F, C, S,Q, dates):
  # Convert C and S to NumPy arrays to perform element-wise operations
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)
    stock_price = (expences_F * (1 + benefit_rate) - u_sale * S + u_price * C) / (C - S) # Stock Price Dynamics

    date_zero = np.where(stock_price <= 0)[0] # Find the dates when the stock price is zero
    if len(date_zero) > 0:
        date_zero = date_zero[0] # Get the first date when the stock price is zero
    else:
        date_zero = None # If there is no date when the stock price is zero, set it to None
        
    stock_price[stock_price <= 0] = 0 # Set the stock price to zero when it is negative

    # Convert the result into a dictionary for JSON response
    data = {
        "start_date": start_date.isoformat(),
        "dates": dates,
        "stock_price": stock_price.tolist(),
        "quantity_ordered": Q,
        "cumulative_supply": C.tolist()
    }

    return data

def StockPrice(expences_F, benefit_rate, u_sale,u_price, S,C ):
    stock_price = (expences_F * (1 + benefit_rate) - u_sale * S + u_price * C) / (C - S) # Stock Price Dynamics
    date_zero = np.where(stock_price <= 0)[0] # Find the dates when the stock price is zero
    if len(date_zero) > 0:
        date_zero = date_zero[0] # Get the first date when the stock price is zero
    else:
        date_zero = None # If there is no date when the stock price is zero, set it to None
    stock_price[stock_price <= 0] = 0 # Set the stock price to zero when it is negative


def StockPriceSimulation(u_price, u_sale, benefit_rate, expences_F, delay_supply, stock_rate, order_n, order_f, due_date, order_q, order_fluc):
    supply = SupplyA(delay_supply, stock_rate, order_n, order_f, due_date, order_q, order_fluc)
    C = supply['SC_Quantity'] # Cumulative Supply Quantity
    S = supply['QuantityOC'] # Cumulative Quantity Ordered
    n_D = len(C) # Number of Dates
    Date = supply['Date'][:n_D] # Dates
    stock_price = (expences_F * (1 + benefit_rate) - u_sale * S + u_price * C) / (C - S) # Stock Price Dynamics
    # Stock-flow speed calculation: f(i)t = S(i)t / A(i)t
    stock_flow_speed = S / C

    # Cost-weight calculation: w(i)t = (A(i)t * a(i)) / sum(A(i)t * a(i))
    total_weight = np.sum(C * u_price)
    cost_weight = (C * u_price) / total_weight

    # Basket calculations
    basket_sale_price = np.sum(u_sale * cost_weight * stock_flow_speed)
    basket_purchase_price = np.sum(u_price * cost_weight * stock_flow_speed)
    stock_basket_price = np.sum(stock_price * cost_weight * stock_flow_speed)    

    date_zero = np.where(stock_price <= 0)[0] # Find the dates when the stock price is zero
    if len(date_zero) > 0:
        date_zero = date_zero[0] # Get the first date when the stock price is zero
    else:
        date_zero = None # If there is no date when the stock price is zero, set it to None
    stock_price[stock_price <= 0] = 0 # Set the stock price to zero when it is negative
    # Calculate the Quantity Ordered and Cumulative Supply Quantity for plotting
    Q = np.insert(np.diff(S), 0, S.iloc[0]) 
    S_cum = np.insert(np.diff(C), 0, C.iloc[0])

    # Convert the result into a dictionary for JSON response
    data = {
        "dates": Date.tolist(),
        "stock_price": stock_price.tolist(),
        "quantity_ordered": Q.tolist(),
        "cumulative_supply": C.tolist(),
        "stock_flow_speed": stock_flow_speed.tolist(),
        "cost_weight": cost_weight.tolist(),
    }
    return jsonify(data)

# an endpoint to request real old data from an excel file
@app.route('/api/realdata', methods=['POST', 'OPTIONS'])
def real_data():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    # read request input
    data = request.get_json()
    benefit_rate = float(data.get('benefit_rate')) # Expected benefit rate
    expences_F = float(data.get('expences_F'))     # Fixed expenses
    unit_sale_price = float(data.get('u_sale')) # Unitary sale price
    unit_purchase_price = float(data.get('u_price')) # Unitary purchase price

    # read data from the excel file
    data = pd.read_excel('data_sheet.xlsx')
    # data = data.to_dict()
    dates = list(data['Dates'])
    cumulative_supply = list(data['Cumulative Supply'].tolist())
    quantity_ordered = list(data['Quantity Ordered'].tolist())
    cumulative_quantity_ordered = list(data['Cumulative Quantity Ordered'].tolist())
    start_date = data['Actual Dates'][0]
    return StockPriceReview(start_date, unit_purchase_price, unit_sale_price, benefit_rate, expences_F, cumulative_supply, cumulative_quantity_ordered,quantity_ordered, dates)

# an endpoint to get simulation data and return the simulation results
@app.route('/api/simulation', methods=['POST', 'OPTIONS'])
def run_simulation():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200  # Respond with 200 OK without doing anything else
    # Read parameters from the request JSON
    data = request.get_json()
 # Validate and convert data
    try:
        u_price = float(data.get('u_price'))          # Unitary price of the stock
        u_sale = float(data.get('u_sale'))            # Unitary sale price
        benefit_rate = float(data.get('benefit_rate')) # Expected benefit rate
        expences_F = float(data.get('expences_F'))     # Fixed expenses
        delay_supply = int(data.get('delay_supply'))    # Delay in replenishing stock
        stock_rate = float(data.get('stock_rate'))      # Stock rate
        order_n = int(data.get('order_n'))              # Expected number of orders
        order_f = int(data.get('order_f'))              # Orders every few days
        due_date = int(data.get('due_date'))            # Simulate for a number of days
        order_q = int(data.get('order_q'))              # Mean order quantity
        order_fluc = float(data.get('order_fluc'))      # Order quantity fluctuation

        
    except (ValueError, TypeError) as e:
        print('Error parsing parameters:', e)
        return jsonify({'error': 'Invalid input data', 'details': str(e)}), 400

    # Call to your StockPriceSimulation function (ensure it handles errors gracefully)
    try:
        return StockPriceSimulation(u_price, u_sale, benefit_rate, expences_F, delay_supply, stock_rate, order_n, order_f, due_date, order_q, order_fluc)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)

