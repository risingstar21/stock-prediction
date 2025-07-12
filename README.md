# 📈 Stock Market Prediction using Linear Regression

This is a simple **Machine Learning + Streamlit** project that predicts **stock prices** using **Linear Regression** on dummy data.

---

## 🧠 Project Story

Imagine you're watching the stock market and wondering,  
*"What will the price be in the next few days?"*

This beginner-friendly project simulates that idea by generating fake stock prices over 100 days. Using those prices, we train a linear regression model to **predict future stock prices** — just like how real-world prediction tools work (on a small scale).

---

## 🚀 Technologies Used

- **Python**
- **Pandas** for handling data
- **NumPy** for numerical operations
- **Scikit-learn** for machine learning (Linear Regression)
- **Matplotlib** for plotting graphs
- **Streamlit** for building a beautiful and interactive web app

---

## 📊 What the App Does

- Shows dummy stock price data (generated randomly but with a trend)
- Trains a Linear Regression model to learn from past stock prices
- Evaluates the model using **Mean Squared Error** and **R² Score**
- Visualizes **actual vs predicted prices**
- Lets the user select a future day (e.g., Day 110) and get a **predicted stock price**

---

## 🖥️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/risingstar21/stock-prediction.git
cd stock-prediction
2. Create a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate    # For Windows
3. Install the required libraries
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
📷 Screenshot Preview

(Add a screenshot of your app later)

🙋 Who is This For?
Beginners in Machine Learning

Students learning Linear Regression

Anyone curious about basic stock prediction models

📌 Future Improvements
Use real-world stock data (from Yahoo Finance or NSE APIs)

Add multiple features like Volume, Opening Price, etc.

Use more advanced models like LSTM or Random Forest

Deploy the app online using Streamlit Cloud

🧑‍💻 Author
Ansh Agrawal
GitHub: risingstar21

⭐ Give a Star!
If you liked this project or learned something new, please ⭐ the repo — it motivates me to build more beginner-friendly projects 😊