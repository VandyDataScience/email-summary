from flask import Flask, request, render_template
from summarization import generate_summary

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        result = generate_summary(user_input)
        return f'The result is: {result}'
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
