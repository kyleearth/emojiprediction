import fasttext as ft
import pandas as pd
import numpy as np
import csv
import time


class we_fasttext:

    def __init__(self, INPUT_PATH, OUTPUT_PATH_TRAIN, OUTPUT_PATH_VALID):
        self.INPUT_PATH = INPUT_PATH
        self.OUTPUT_PATH_TRAIN = OUTPUT_PATH_TRAIN
        self.OUTPUT_PATH_VALID = OUTPUT_PATH_VALID
        self.df = pd.read_csv(INPUT_PATH, header=0)
        self.length = self.df.shape[0]
        self.model = None

    def print_results(self, N, p, r, k):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(k, p))
        print("R@{}\t{:.3f}".format(k, r))
        print("f1@{}\t{:.3f}".format(k, (2 * p * r) / (p + r)))

    def getRecallAtK(self, test: str):
        recall = []
        for i in range(1, 16):
            N, p, r = self.model.test(test, k=i)
            recall.append(r)
        return recall

    def train_txt(self, size=0.8):
        file = open(self.OUTPUT_PATH_TRAIN, 'w+')
        for i in range(int(self.length * size)):
            line = '__label__' + str(self.df['label'].iloc[i]) +\
                ' ' + str(self.df['filtered_tweet'].iloc[i])
            file.write(line + '\n')

    def valid_txt(self, size=0.2):
        file = open(self.OUTPUT_PATH_VALID, 'w+')
        for i in range(int(self.length * (1 - size)), self.length):
            line = '__label__' + str(self.df['label'].iloc[i]) +\
                ' ' + str(self.df['filtered_tweet'].iloc[i])
            file.write(line + '\n')

    def train_valid_ori_txt(self):

        training_time = time.time()

        # epoch: number of epoch
        # lr: learning rate
        # wordNgrams: bags of words
        # loss: 'ova' is one vs all for multiclassification
        model = ft.train_supervised(
            self.OUTPUT_PATH_TRAIN, epoch=5, lr=0.1, wordNgrams=2, loss='ova')
        # model = ft.train_supervised(
        #     input=self.OUTPUT_PATH_TRAIN, autotuneValidationFile=self.OUTPUT_PATH_VALID)
        self.model = model
        # k: compute the precision at k and recall at k
        print('Training time is %.3f seconds' % (time.time() - training_time))

        validating_time = time.time()
        self.print_results(*model.test(self.OUTPUT_PATH_VALID))
        print('Validating time is %.3f seconds' %
              (time.time() - validating_time))

    def getRP_by_label(self, test):
        mapping_label = self.model.test_label(test)

        # get mapping list corresponding to emoji
        mapping = [l.strip().split()
                   for l in open('mapping.txt', encoding='utf8')]
        map_dict = {}
        for m in mapping:
            map_dict[m[0]] = m[1]

        # get corresponing p, r and f1
        for k, v in mapping_label.items():
            label = k.replace('__label__', '')
            p = v['precision']
            r = v['recall']
            f1 = v['f1score']
            print(map_dict[label], ': precision is %.5f; recall is %.5f; f1 score is %.5f' %
                  (p, r, f1))

    def train_valid_txt(self, train_valid, test, k):

        training_time = time.time()

        # epoch: number of epoch
        # lr: learning rate
        # wordNgrams: bags of words
        # loss: 'ova' is one vs all for multiclassification

        model = ft.train_supervised(
            train_valid,
            epoch=14,
            lr=0.1,
            wordNgrams=4,
            loss='ova',
            min_count=1)
    # model = ft.train_supervised(
    #     input=self.OUTPUT_PATH_TRAIN, autotuneValidationFile=self.OUTPUT_PATH_VALID)
        self.model = model
        # k: compute the precision at k and recall at k
        print('Training time is %.3f seconds' %
              (time.time() - training_time))

        validating_time = time.time()
        self.print_results(*model.test(test, k=k), k=k)
        print('Validating time is %.3f seconds' %
              (time.time() - validating_time))

    def getModel(self):
        return self.model

    def getPredictRes(self, test):
        res = []
        df = pd.read_csv(test)
        for i in range(df.shape[0]):
            if df.iloc[i]['filtered_tweet'] is np.nan:
                continue
            res.append(self.model.predict(df.iloc[i]['filtered_tweet'], k=15))
        prob_list = pd.DataFrame()

    def getPred(self, test):
        res = []
        df = pd.read_csv(test)
        for i in range(df.shape[0]):
            if df.iloc[i]['filtered_tweet'] is np.nan:
                continue
            temp = self.model.predict(df.iloc[i]['filtered_tweet'], k=1)
            temp = temp[0][0].replace('__label__', '')
            res.append(temp)
        preds = pd.DataFrame()
        preds['y_pred'] = res
        preds['y_true'] = df['label']
        preds.to_csv('fastttext_preds.csv', index=False)


if __name__ == '__main__':

    INPUT_PATH = './Data/clean_data.csv'
    OUTPUT_PATH_TRAIN = './Data/ft_train.txt'
    OUTPUT_PATH_VALID = './Data/ft_valid.txt'

    emft = we_fasttext(INPUT_PATH, OUTPUT_PATH_TRAIN, OUTPUT_PATH_VALID)
    emft.train_txt()
    emft.valid_txt()
    # emft.train_valid_ori_txt()

    emft.train_valid_txt(
        train_valid='./Data/train_valid.txt', test='./Data/test.txt', k=1)
    print(emft.getModel().predict('asdfdasfadsfadsf', k=15)[1])
    # emft.getPred(test='./Data/test_set.csv')
    # print(emft.getModel().predict('asdfasdf', k=1)[0][0])
    # recall = emft.getRecallAtK(test='./Data/test.txt')
    # df = pd.DataFrame(recall, columns=['recall_ft'])
    # df.to_csv('recall_ft.csv')
    # emft.getRP_by_label(test='./Data/test.txt')
    # matrix = emft.getModel().get_input_matrix()
    # print(matrix)
    # print(matrix.shape)
