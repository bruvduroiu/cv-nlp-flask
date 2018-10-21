from services.nlp import WordCluster

if __name__ == '__main__':
    wc = WordCluster(dataset_path='data/data_6.json', json=True)

    print(wc.get_topic_json(['aberqerfiqowjefoiuqergqieurhg']))