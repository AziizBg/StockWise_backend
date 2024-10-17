import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def quantityO(n, lambd, d, m, sigma):
    D = datesO(n, lambd)
    k = beforeD(d, D)
    D = D[:k]
    Q = np.random.binomial(n * lambd, 1 - sigma**2 / m, k)
    Q = np.cumsum(Q)
    Q = np.round(Q)
    out = pd.DataFrame({'Date': D, 'Quantity': np.insert(np.diff(Q), 0, Q[0]), 'C_Quantity': Q})
    return out

# Function to provide a supply sequence based on the sales sequence
def SupplyA(delay, alpha, n, lambd, d, m, sigma):
    sales = quantityO(n, lambd, d, m, sigma)
    S1 = sales['Date'] - delay
    S2 = np.round(sales['Quantity'] * (1 + alpha))
    S3 = np.cumsum(S2)
    outsupply = pd.DataFrame({'S_Dates': S1, 'SC_Quantity': S3, 'Date': sales['Date'], 'QuantityOC': sales['C_Quantity']})
    return outsupply

# Stock Price Dynamics function
def StockPriceD(u_price, u_sale, benefit_rate, expences_F, delay_supply, stock_rate, order_n, order_f, due_date, order_q, order_fluc):
    supply = SupplyA(delay_supply, stock_rate, order_n, order_f, due_date, order_q, order_fluc)
    C = supply['SC_Quantity']
    S = supply['QuantityOC']
    n_D = len(C)
    Date = supply['Date'][:n_D]
    
    stock_price = (expences_F * (1 + benefit_rate) - u_sale * S + u_price * C) / (C - S)
    
    date_zero = np.where(stock_price <= 0)[0]
    if len(date_zero) > 0:
        date_zero = date_zero[0]
    else:
        date_zero = None
        
    stock_price[stock_price <= 0] = 0
    
    Q = np.insert(np.diff(S), 0, S.iloc[0])
    S_cum = np.insert(np.diff(C), 0, C.iloc[0])
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Forecast Sales
    plt.subplot(1, 3, 1)
    plt.step(supply['Date'], Q, where='mid', label='Quantity Ordered', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Quantity Ordered')
    plt.title('Forecast Sales')
    
    # Plot 2: Forecast Supply Planning
    plt.subplot(1, 3, 2)
    plt.step(supply['S_Dates'], supply['SC_Quantity'], where='mid', label='Cumulative Supply Quantity', color='red')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Supply Quantity')
    plt.title('Forecast Supply Planning')
    
    # Plot 3: Stock Price Dynamics
    plt.subplot(1, 3, 3)
    plt.step(Date, stock_price, where='mid', label='Stock Price', color='blue')
    plt.axhline(y=u_sale, color='blue', linestyle='--', label='Unit Sale Price')
    plt.axhline(y=u_price, color='green', linestyle='--', label='Unit Purchase Price')
    
    if date_zero is not None:
        plt.axvline(x=Date.iloc[date_zero], color='red', linestyle='--', label='Stock Price Zero Date')
    
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Dynamics')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("S:", S)
    print("=====================================")
    print("C:", C)
    print("=====================================")
    print("Q:", Q.tolist())

# Example parameters to test
u_price = 50          # Unitary price of the stock
u_sale = 100          # Unitary sale price
benefit_rate = 0.1    # Expected benefit rate (10%)
expences_F = 10000    # Fixed expenses (e.g., $10,000)
delay_supply = 5      # 5-day delay in replenishing stock
stock_rate = 0.2      # 20% stock security rate
order_n = 50          # Expected number of orders
order_f = 4           # Orders every 4 days (Poisson distributed)
due_date = 60         # Simulate for a 60-day period
order_q = 100         # Mean order quantity
order_fluc = 5        # Order quantity fluctuation (standard deviation)

# Run the function and see the plots
StockPriceD(u_price, u_sale, benefit_rate, expences_F, delay_supply, stock_rate, order_n, order_f, due_date, order_q, order_fluc)
