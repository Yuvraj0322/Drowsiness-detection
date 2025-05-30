from flask import Flask, render_template, jsonify
import threading
import os
import sys

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/start_detection')
def start_detection_route():
    print("Start detection route called")
    from drowsiness_detection import start_detection
    threading.Thread(target=start_detection, daemon=True).start()  # Run in background
    return jsonify({"message": "Drowsiness detection started."})


if __name__ == '__main__':
    app.run(debug=True)
