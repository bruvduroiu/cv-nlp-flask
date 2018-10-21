from services.nlp import WordCluster

if __name__ == '__main__':
    wc = WordCluster(dataset_path='data/data_3.json', json=True, retrain=True)

    print(wc.get_topic_json(['aberqerfiqowjefoiuqergqieurhg']))