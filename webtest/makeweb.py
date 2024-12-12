from flask import Flask, render_template, request
import os
import signal

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    # Flask 애플리케이션을 종료하는 함수
    os.kill(os.getpid(), signal.SIGINT)
    return "서버가 종료되었습니다."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)