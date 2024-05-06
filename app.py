from flask import Flask, render_template, request, jsonify
import pandas as pd
from datascrap import Webscrapper , callPredict , callResult

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        result = Webscrapper(url)
        # ส่ง url ไปยัง Webscrapper และใช้ผลลัพธ์ที่ได้กลับมา
        return 'URL Scraping Result: ' + str(result)  # แสดงผลลัพธ์ที่ได้
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url', '')
    callPredict(url)
    return jsonify({'result': 'Prediction called'})

@app.route('/result', methods=['GET'])
def result():
    result_code = callResult()
    return render_template('result.html', result=result_code) 

if __name__ == '__main__':
    app.run(debug=True)