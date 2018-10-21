import json

from flask import Flask

from services.nlp import WordCluster

app = Flask(__name__)


@app.route('/clusters', methods=['GET'])
def clusters():
    return json.dumps(wc.get_topic_json(['Soy muy precioso']))


@app.route('/')
def hello_world():
    return 'Hello world'


if __name__ == '__main__':
    wc = WordCluster(n_clusters=5, dataset_path='data/data_3.json', json=True)
    app.run()