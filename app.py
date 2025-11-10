import os
import pandas as pd
import sqlite3
import pickle
from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    # Expanded, higher-quality training dataset
    data = {
        'description': [
            # Food
            'pizza', 'burger', 'restaurant', 'cafe', 'dinner', 'breakfast', 'snack', 'lunch', 'coffee',
            # Travel
            'bus', 'uber', 'ola', 'flight', 'train', 'taxi', 'petrol', 'metro', 'cab fare',
            # Shopping
            'shirt', 'jeans', 'shoes', 'bag', 'watch', 'clothes', 'jacket', 'tshirt', 'accessories',
            # Entertainment
            'movie', 'concert', 'netflix', 'spotify', 'cinema', 'game', 'theatre', 'subscription',
            # Utilities
            'electricity bill', 'water bill', 'gas', 'internet', 'mobile recharge', 'rent', 'insurance',
            # Groceries
            'milk', 'bread', 'vegetables', 'fruits', 'groceries', 'supermarket', 'rice', 'oil'
        ],
        'category': [
            # Food
            'Food', 'Food', 'Food', 'Food', 'Food', 'Food', 'Food', 'Food', 'Food',
            # Travel
            'Travel', 'Travel', 'Travel', 'Travel', 'Travel', 'Travel', 'Travel', 'Travel', 'Travel',
            # Shopping
            'Shopping', 'Shopping', 'Shopping', 'Shopping', 'Shopping', 'Shopping', 'Shopping', 'Shopping', 'Shopping',
            # Entertainment
            'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment',
            # Utilities
            'Utilities', 'Utilities', 'Utilities', 'Utilities', 'Utilities', 'Utilities', 'Utilities',
            # Groceries
            'Groceries', 'Groceries', 'Groceries', 'Groceries', 'Groceries', 'Groceries', 'Groceries', 'Groceries'
        ]
    }

    df = pd.DataFrame(data)

    # TF-IDF: smarter word weighting
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['description'])

    # Logistic Regression for better accuracy
    model = LogisticRegression(max_iter=1000)
    model.fit(X, df['category'])

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((vectorizer, model), f)

    print("✅ New ML model trained and saved.")
else:
    # Load pre-trained model
    with open(MODEL_PATH, 'rb') as f:
        vectorizer, model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend (fixes warning)
    import matplotlib.pyplot as plt
    import io, base64
    file = request.files['file']
    if not file:
        return "No file uploaded!"

    df = pd.read_csv(file)
    if 'description' not in df.columns or 'amount' not in df.columns:
        return "CSV must have 'description' and 'amount' columns!"

    df['description'] = df['description'].astype(str)
    X_test = vectorizer.transform(df['description'])
    df['category'] = model.predict(X_test)

    # --- Save to DB ---
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql('expenses', conn, if_exists='append', index=False)

    # --- Insights ---
    insights_df = df.groupby('category')['amount'].sum().reset_index()
    top_category = insights_df.loc[insights_df['amount'].idxmax()]
    insights_text = f"Highest spending category: {top_category['category']} (₹{top_category['amount']})"

    # --- Pie Chart (Category Distribution) ---
    plt.figure(figsize=(5,5))
    plt.pie(insights_df['amount'], labels=insights_df['category'], autopct='%1.1f%%', startangle=140)
    plt.title('Spending by Category')

    pie_img = io.BytesIO()
    plt.savefig(pie_img, format='png', bbox_inches='tight')
    pie_img.seek(0)
    pie_url = base64.b64encode(pie_img.getvalue()).decode()
    plt.close()

    # --- Bar Chart (Category vs Amount) ---
    plt.figure(figsize=(6,4))
    plt.bar(insights_df['category'], insights_df['amount'])
    plt.xlabel('Category')
    plt.ylabel('Amount (₹)')
    plt.title('Spending Summary')

    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png', bbox_inches='tight')
    bar_img.seek(0)
    bar_url = base64.b64encode(bar_img.getvalue()).decode()
    plt.close()

    return render_template(
        'result.html',
        table=df.to_html(classes='data', index=False),
        insights=insights_df.to_dict(orient='records'),
        insights_text=insights_text,
        pie_chart=pie_url,
        bar_chart=bar_url
    )

 

if __name__ == '__main__':
    app.run(debug=True)
