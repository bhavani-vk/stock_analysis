#importing required libraries
#importing csv library to read the input files and write the output into text file.
import csv
#importing datetime to handle date operations.
from datetime import datetime
#importing matplotlib to plot line graph
from matplotlib import pyplot as plt
#importing psycopg2 for connecting to the Database
import psycopg2
from psycopg2 import sql
#importing json to read json data that contains stock details.
import json
#importing pandas to read data from db and fo statstical operations.
import pandas as pd
#importing plotly to plot graphs.
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#importing sqlalchemy to connect to db while using pandas.
from sqlalchemy import create_engine
#importing to open the stock and bond csv file.
import tkinter as tk
from tkinter import filedialog
#importing yfinance to get the latest stock details.
import yfinance as yf
#importing seaborn to plot corelation of each stock.
import seaborn as sns

# Creating DBConnection to open the connection between our script and database.
class DBConnection:

    def get_url():
        return "postgresql://username:password@localhost:5432/stocks"

    @staticmethod
    def conn():
        try:
            #Establishing the connection
            conn = psycopg2.connect(DBConnection.get_url())
            return conn
        except Exception as e:
            print(f'Error connecting to database to validate Token: {e}')

class Investor:
    #creating invertor class.
    def __init__(self, investor_id, name):
        try:
            #initilizing the inverstor data variables
            self.investor_id = investor_id
            self.name = name
        except Exception as e:
            print(f"error initializing Investor data: {e}")

class Stock:
    #creating stock class.
    def __init__(self, purchase_id, stock_symbol, no_of_shares, purchase_price, current_price, purchase_date):
        try:
            #initilizing the stock data variables
            self.purchase_id = purchase_id
            self.stock_symbol = stock_symbol
            self.no_of_shares = no_of_shares
            self.purchase_price = purchase_price
            self.current_price = current_price
            self.purchase_date = purchase_date
            self.earn_or_loss = self.get_earning_or_loss()
            self.year_earn_or_loss = self.get_earning_or_loss(datetime.today())
        except Exception as e:
            print(f"error initializing stock data: {e}")

    #Function to calculate earning and loss and yearly earning and loss using Method Overloading
    def get_earning_or_loss(self, current_date=None):
        try:
            #checking the the condition if current date passed in parameter it calculates yearly earning/loss if not jut earning/loss of the stock to the current price.
            if current_date is None:
                #calculating earning/loss and returning the value 
                return round(((self.current_price - self.purchase_price) * self.no_of_shares), 2)
            else:
                #calculating yearly earning/loss and returning the value 
                no_of_date_change = current_date.date() - self.purchase_date
                return round((((self.current_price - self.purchase_price) / self.purchase_price) / (no_of_date_change.days/365)) * 100, 2)
        except Exception as e:
            print(f"error calculating earning_or_loss: {e}")

class Bond(Stock):
    def __init__(self, purchase_id, stock_symbol, no_of_shares, purchase_price, current_price, purchase_date, coupon, yield_rate):
        try:
            #inheriting the stock variable which are common in both stock and bond.
            super().__init__(purchase_id, stock_symbol, no_of_shares, purchase_price, current_price, purchase_date)
            #initilizing the bond data variables
            self.coupon = coupon
            self.yield_rate = yield_rate
        except Exception as e:
            print(f"error initializing bond data: {e}")

def main():
    try:
        #setting up data in database if data doesn't exist
        create_database()
        #inserting data if not available in database
        create_table()
        #creating investor object
        investor = get_investor_data()
        #calling get_stock_data to get the stock data. all data are recevied as array of stock object and stored in stocks
        stocks, shares = get_stock_data()
        #calling get_bond_data to get the bond data. all data are recevied as array of bond object and stored in bond
        bonds = get_bond_data()
        #ploting a graph price by date of stock all stock in one image.
        show_graph(shares)
        #ploting a graph price by date of stock each stock
        show_additional_graph(shares)
        #creating output file to store the report.
        output = open('output/output.txt','w')
        #calling print_stock to store stock report in output file
        print_stock(output, stocks, investor)
        #calling print_bond to store bond report in output file
        print_bond(output,bonds,investor)
        for symbol in shares.items():
            latest = fetch_latest_stock_price(symbol[0])
            print(f"latest price of {symbol[0]} = {latest}")
    except Exception as e:
        print(f"error in main function: {e}")

#fetching the stock data from database.
def get_stock_data():
    try:
        try:
            #Using DB connection fetching all the stock data
            db = DBConnection.conn()
            cur = db.cursor()
            #executing the query
            cur.execute("SELECT * FROM stocks_data")
            result = cur.fetchall()
            cur.execute("SELECT stock_symbol,no_of_shares FROM stocks_data")
            share_result = cur.fetchall()
            #closing the connection
            cur.close()
            db.close()
        except Exception as e:
            print(f"error importing csv file: {e}")
        if result is None:
            #validating if data is not empty
            print("No stock data present in csv")
            return []
        stock = []
        for res in result:
            #using a loop for each data creating an stock object.
            stock.append(Stock(res[0],res[1],res[2],res[3],res[4],res[5]))
        shares = {}
        for res in share_result:
            shares[res[0]] = res[1]
        return stock,shares
    except Exception as e:
            print(f"error importing stock data: {e}")

#fetching the bond data from database.
def get_bond_data():
    try:
        try:
            #Using DB connection fetching all the bond data
            db = DBConnection.conn()
            cur = db.cursor()
            #executing the query
            cur.execute("SELECT * FROM bonds_data")
            result = cur.fetchall()
            #closing the connection
            cur.close()
            db.close()
        except Exception as e:
            print(f"error importing csv file: {e}")
        if result is None:
            #validating if data is not empty
            print("No bond data present in csv")
            return []
        bond = []
        for res in result:
            #using a loop for each data creating an bond object.
            bond.append(Bond(res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7]))
        return bond
    except Exception as e:
            print(f"error importing bond data: {e}")
#fetching the investor data from database.
def get_investor_data():
    try:
        try:
            #Using DB connection fetching all the investor data
            db = DBConnection.conn()
            cur = db.cursor()
            #executing the query
            cur.execute("SELECT * FROM investor")
            result = cur.fetchall()
            #closing the connection
            cur.close()
            db.close()
        except Exception as e:
            print(f"Failed importing investor data from Database: {e}")
        #validating if data is not empty
        if result is None:
            print("No bond investor present in csv")
            return []
        investor = []
        for res in result:
            investor.append(Investor(res[0],res[1]))
        return investor
    except Exception as e:
            print(f"error importing investor data: {e}")

def print_stock(output,stocks,investor):
    try:
        #printing all stocks
        line = ("-" * 73) + "\n"
        output.write(line)
        #print investor name.
        output.write(f"| Stock ownership for {investor[0].name:49s} |\n")
        output.write(line)
        # Used sting formatting given width specifier to print report "{no of character space}s" 
        #printing header for stock report
        output.write(f"| {'Stock':15s}| {'Share #':14s}| {'Earnings/Loss':13s}| {'Yearly Earning/Loss':22s}|\n")
        output.write(line)
        for stock in stocks:
            #using for printing all the stock data
            output.write(f"| {stock.stock_symbol:15s}| {stock.no_of_shares}\t\t| ${stock.earn_or_loss:4.2f}\t| {stock.year_earn_or_loss:4.2f}{'%':9s} \t|\n")
            output.write(line)
    except Exception as e:
            print(f"error printing stocks: {e}")

def print_bond(output,bonds,investor):
    try:
        # Printing all bonds
        line = ("-" * 162) + "\n"
        output.write(f"\n{line}")
        output.write(f"| Bond ownership for {investor[0].name:139} |\n")
        output.write(line)
        # Used sting formatting given width specifier to print report "{no of character space}s" 
        # printing header for bond report
        output.write(f"| {'Bond':14s} | {'Share #':13s} | {'Purchase Price':21s} | {'Current Price':21s} | {'Coupon':13s} | {'Yield':2s} | {'Purchase Date':12s} | {'Earnings/Loss':12s} | {'Yearly Earning/Loss':21s} |\n" )
        output.write(line)
        for bond in bonds:
            #using for printing all the bond data
            output.write(f"| {bond.stock_symbol}\t | {bond.no_of_shares}\t\t | ${bond.purchase_price}\t\t | ${bond.current_price}\t\t | {bond.coupon}\t\t | {bond.yield_rate}\t | {bond.purchase_date}\t | ${bond.earn_or_loss}\t\t | {bond.year_earn_or_loss}{'%'}\t\t\t |\n")
            output.write(line)
    except Exception as e:
            print(f"error printing bonds: {e}")

def create_table():
    try:
        #call connection
        db = DBConnection.conn()
        cur = db.cursor()
        #queries to create table is not exists
        table_queries = [
            """
            CREATE TABLE IF NOT EXISTS investor (
                investor_id character varying(64) NOT NULL,
                name character varying(256) NOT NULL,
                CONSTRAINT investor_pkey PRIMARY KEY (investor_id),
                CONSTRAINT investor_unique_key UNIQUE (investor_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS stocks_data(
                stock_id character varying(64) NOT NULL UNIQUE,
                stock_symbol character varying(64) NOT NULL,
                no_of_shares integer NOT NULL,
                purchase_price float NOT NULL,
                current_price float NOT NULL,
                purchase_date date NOT NULL,
                CONSTRAINT stocks_pkey PRIMARY KEY (stock_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS bonds_data(
                bond_id  character varying(64)  NOT NULL UNIQUE,
                stock_symbol character varying(64)  NOT NULL,
                no_of_shares integer NOT NULL,
                purchase_price float NOT NULL,
                current_price float NOT NULL,
                purchase_date date NOT NULL,
                coupon float NOT NULL,
                yield_rate float NOT NULL,
                CONSTRAINT bond_pkey PRIMARY KEY (bond_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS stocks_info(
                stock_id character varying(64) NOT NULL UNIQUE,
                symbol character varying(64) NOT NULL,
                date date NOT NULL,
                open character varying(64) NOT NULL,
                high character varying(64) NOT NULL,
                low character varying(64) NOT NULL,
                close float NOT NULL,
                volume integer NOT NULL,
                CONSTRAINT stock_primary_key PRIMARY KEY (stock_id)
            );
            """
        ]
        #executing create table query
        for query in table_queries:
            cur.execute(query)
        #commiting the process to avoid loss of data
        db.commit()

        #inserting into investor table
        investor_query = "INSERT INTO investor (investor_id, name) VALUES (%s, %s) ON CONFLICT (investor_id) DO NOTHING"
        investor_data = ('INV001','Bob Smith')
        #executing the query
        cur.execute(investor_query, investor_data)
        #commiting the process to avoid loss of data
        db.commit()

        #opening the stock csv file
        stockfile  = filedialog.askopenfilename(title="Select Stock CSV File", filetypes=[("CSV Files", "*.csv")])
        #opening bond csv file
        bondfile = filedialog.askopenfilename(title="Select Bond CSV File", filetypes=[("CSV Files", "*.csv")])

        #reading the data and adding to the list from csv
        stock_data = list(csv.reader(open(stockfile)))
        #reading the data
        bond_data = list(csv.reader(open(bondfile)))

        stock_id = "STK"
        #inserting stock data to database
        stock_query = """INSERT INTO stocks_data (stock_id, stock_symbol, no_of_shares, purchase_price, current_price, purchase_date)
                    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (stock_id) DO NOTHING"""
        for i in range(1, len(stock_data)):
            data = ((stock_id+str(i)), stock_data[i][0], int(stock_data[i][1]), float(stock_data[i][2]), float(stock_data[i][3]), stock_data[i][4])
            #executing the query
            cur.execute(stock_query, data)
            #commiting the process to avoid loss of data
            db.commit()

        bond_id = "BOND"
        #inserting bond data to database
        bond_query = """INSERT INTO bonds_data(
        bond_id, stock_symbol, no_of_shares, purchase_price, current_price, purchase_date, coupon, yield_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (bond_id) DO NOTHING"""
        for i in range(1, len(bond_data)):
            data = ((bond_id+str(i)), bond_data[i][0], int(bond_data[i][1]), float(bond_data[i][2]), float(bond_data[i][3]), bond_data[i][4], float(bond_data[i][5]), float(bond_data[i][6]))
            #executing the query
            cur.execute(bond_query, data)
            #commiting the process to avoid loss of data
            db.commit()

        #reading json file.
        file = open("data/AllStocks.json")
        #loading the json data.
        stock_info = json.load(file)
        #query to inserting json data to the database.
        stock_info_query = """
            INSERT INTO public.stocks_info(
            stock_id, symbol, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (stock_id) DO NOTHING;
        """
        i = 1
        for stock in stock_info:
            #using loop inserting all the data to the database.
            data = (i, stock['Symbol'], stock['Date'], stock['Open'], stock['High'], stock['Low'], stock['Close'], stock['Volume'])
            i += 1
            #executing query
            cur.execute(stock_info_query, data)
            #commiting the process to avoid loss of data
            db.commit()

        #closing the connection
        cur.close()
        db.close()
    except Exception as e:
        print(f"error while creating table or inserting data :  {e}")

def create_database():
    try:
        dbname = "stocks"
        connection = psycopg2.connect("postgresql://username:password@localhost:5432")
        connection.autocommit = True  # Needed to create a database
        cursor = connection.cursor()

        # Check if the database exists
        cursor.execute(sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s"), [dbname])
        exists = cursor.fetchone()

        # Create the database if it doesn't exist
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
            print(f"Database '{dbname}' created successfully.")
        else:
            print(f"Database '{dbname}' already exists.")

        #Close the connection to the default database
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")

def show_graph(shares):
    try:
        #get db url
        db_url = DBConnection.get_url()
        #opening connection
        engine = create_engine(db_url)
        # Query to retrieve stock data
        query = """
            SELECT symbol, date, close
            FROM stocks_info
            ORDER BY symbol, date;
        """
        # Execute the query and load the data into a pandas DataFrame
        df = pd.read_sql(query, engine)
        # Convert the Date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        grouped = df.groupby('symbol')
        # Plot the graph
        stock_values = {}
        for symbol, group in grouped:
            if symbol in shares:
                stock_values[symbol] = group[['date', 'close']].copy()
                stock_values[symbol]['Value'] = stock_values[symbol]['close'] * shares[symbol]

        fig = make_subplots(rows=1, cols=1)
        for symbol, data in stock_values.items():
            fig.add_trace(go.Scatter(x=data['date'], y=data['Value'], mode='lines', name=symbol))

        # Save the plot to a file
        fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Portfolio Value")
        fig.write_html("output/PortfolioValue.html")
        print("Graph saved as PortfolioValue.html")
    except Exception as error:
        print(f"Error while ploting graph: {error}")

def show_additional_graph(shares):
    try:
        #get db url
        db_url = DBConnection.get_url()
        #opening connection
        engine = create_engine(db_url)
        # Query to retrieve stock data
        query = """
            SELECT symbol, date, open, high, low, close
            FROM stocks_info
            ORDER BY symbol, date;
        """
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])
        grouped = df.groupby('symbol')
        # Plot the graph
        stock_values = {}
        for symbol, group in grouped:
            if symbol in shares:
                stock_values[symbol] = group[['date', 'open', 'high', 'low', 'close']].copy()

        for symbol, data in stock_values.items():
            #using loop ploting graph for each stocks
            fig = go.Figure(data=[go.Candlestick(x=data['date'],
                                                open=data['open'],
                                                high=data['high'],
                                                low=data['low'],
                                                close=data['close'])])
            fig.update_layout(title=f"Price Movements for {symbol} Over Time", xaxis_title="Date", yaxis_title="Price")
            # Save the plot to a file
            fig.write_html(f"output/{symbol}_PriceMovements.html")
            print(f"Price Movement Graph saved as {symbol}_PriceMovements.html")
    except Exception as e:
        print(f"Error while ploting additional graph: {e}")

def fetch_latest_stock_price(symbol):
    try:
        #geting the latest price of the stock.
        stock = yf.Ticker(symbol)
        return round(stock.history(period="1d")['Close'].iloc[-1],2)
    except:
        return "No data found, symbol may be delisted"


def pandas_analysis():
    #get db url
    db_url = DBConnection.get_url()
    #opening connection
    engine = create_engine(db_url)
    query = "SELECT * FROM stocks_info"
    df = pd.read_sql(query, engine)

    # Example analysis: Calculate moving averages
    df['date'] = pd.to_datetime(df['date'])
    df['50_day_MA'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
    df['200_day_MA'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=200).mean())

    df.to_csv('output/stocks_moving_averages.csv', index=False)
    print("Moving averages saved to stocks_moving_averages.csv")

def seaborn_correlation_heatmap():
    try:
        #get db url
        db_url = DBConnection.get_url()
        #opening connection
        engine = create_engine(db_url)
        query = """
            SELECT symbol, date, close
            FROM stocks_info
            ORDER BY symbol, date;
        """
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])

        # Pivot the data to create a matrix of stock symbols and their closing prices
        pivot_df = df.pivot(index='date', columns='symbol', values='close')

        # Calculate the correlation matrix
        corr_matrix = pivot_df.corr()

        # Plotting the heatmap
        sns.set(style="white")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5)
        plt.title("Correlation Heatmap of Stock Prices")

        # Save the heatmap to a file
        output_file = 'output/correlation_heatmap.png'
        plt.savefig(output_file)
        print(f"Correlation heatmap saved as {output_file}")
    except Exception as e:
        print(f"error creating seaborn heatmap: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    # Hide the root window
    root.withdraw()
    #calling main function
    main()
    #calculating moving average
    pandas_analysis()
    #corelation between each stock
    seaborn_correlation_heatmap()
