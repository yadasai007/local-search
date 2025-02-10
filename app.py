from flask import Flask, render_template, request
from local_data import search_value
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    print(search_value(user_input))
    results=search_value(user_input)
    return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run()
