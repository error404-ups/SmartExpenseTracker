import os
import pandas as pd
import sqlite3
import pickle
from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# --- Database Setup ---
DB_PATH = 'data/expenses.db'
os.makedirs('data', exist_ok=True)

# Create DB table if not exists
with sqlite3.connect(DB_PATH) as conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            description TEXT,
            amount REAL,
            category TEXT
        )
    ''')

# --- Train or Load Model ---
MODEL_PATH = 'model/expense_model.pkl'
os.makedirs('model', exist_ok=True)

if not os.path.exists(MODEL_PATH):
    # Sample training data for simple categorization
    data = {
        'description': [
            'pizza', 'burger', 'restaurant', 'bus', 'uber', 'flight',
            'shirt', 'jeans', 'shoes', 'movie', 'concert', 'electricity bill'
        ],
        'category': [
            'Food', 'Food', 'Food', 'Travel', 'Travel', 'Travel',
            'Shopping', 'Shopping', 'Shopping', 'Entertainment', 'Entertainment', 'Utilities'
        ]
    }
    df = pd.DataFrame(data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['description'])
    model = MultinomialNB()
    model.fit(X, df['category'])

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((vectorizer, model), f)
else:
    with open(MODEL_PATH, 'rb') as f:
        vectorizer, model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file uploaded!"

    df = pd.read_csv(file)
    if 'description' not in df.columns or 'amount' not in df.columns:
        return "CSV must have 'description' and 'amount' columns!"

    df['description'] = df['description'].astype(str)
    X_test = vectorizer.transform(df['description'])
    df['category'] = model.predict(X_test)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql('expenses', conn, if_exists='append', index=False)

    # Basic insight: average spending per category
    insights = df.groupby('category')['amount'].sum().reset_index()

    return render_template(
    'result.html',
    table=df.to_html(classes='data', index=False),
    insights=insights.to_dict(orient='records')
)

 

if __name__ == '__main__':
    app.run(debug=True)
