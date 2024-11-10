import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    st.title("Stock Analysis Web Application")

    # Getting user input for years and stock name
    howmanyyears = st.sidebar.number_input("How many years?", min_value=1, max_value=20, value=5, step=1)
    whichstock = st.sidebar.text_input("Which stock? (e.g., AAPL)", value="AAPL")

    if st.sidebar.button("Analyze"):
        today = date.today()
        END_DATE = today.isoformat()
        START_DATE = date(today.year - howmanyyears, today.month, today.day).isoformat()

        data = yf.download(whichstock, start=START_DATE, end=END_DATE)

        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data.Date)

        if not data.empty:
            st.subheader(f"Data for {whichstock} from {START_DATE} to {END_DATE}")
            st.write(data)

            data['EMA-50'] = data['Close'].ewm(span=50, adjust=False).mean()
            data['EMA-200'] = data['Close'].ewm(span=200, adjust=False).mean()

            # High vs Low Graph
            st.subheader("High vs Low Graph")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data['Low'], label="Low", color="indianred")
            ax.plot(data['High'], label="High", color="mediumseagreen")
            ax.set_ylabel('Price (in USD)')
            ax.set_xlabel("Time")
            ax.set_title(f"High vs Low of {whichstock}")
            ax.legend()
            st.pyplot(fig)

            # Exponential Moving Average Graph
            st.subheader("Exponential Moving Average Graph")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data['EMA-50'], label="EMA for 50 days")
            ax.plot(data['EMA-200'], label="EMA for 200 days")
            ax.plot(data['Adj Close'], label="Close")
            ax.set_title(f'Exponential Moving Average for {whichstock}')
            ax.set_ylabel('Price (in USD)')
            ax.set_xlabel("Time")
            ax.legend()
            st.pyplot(fig)

            x = data[['Open', 'High', 'Low', 'Volume', 'EMA-50', 'EMA-200']].fillna(0)
            y = data['Close']

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            pred = lr_model.predict(X_test)

            d = pd.DataFrame({'Actual_Price': y_test.values.ravel(), 'Predicted_Price': pred.ravel()})
            st.subheader("Predicted vs Actual Price")
            st.write(d.head(10))
            st.subheader("Descriptive Statistics")
            st.write(d.describe())
        else:
            st.error(f"No data found for {whichstock}.")


if __name__ == "__main__":
    main()