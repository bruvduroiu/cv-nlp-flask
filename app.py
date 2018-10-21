from flask import Flask
app = Flask(__name__)


@app.route('/clusters', methods=['GET'])
def clusters():
    return 'Clusters'


@app.route('/')
def hello_world():
    return 'Hello world'