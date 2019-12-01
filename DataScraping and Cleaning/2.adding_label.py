import pandas as pd
import numpy as np
import re
import unicode_codes
import os


class labeling_tweets():

    def combine_files(self, in_folder='./Data/'):
        print('>>> Start to combine json in Data directory:')
        pattern = '.json'

        # list of files in directory
        dirs = os.listdir(in_folder)

        new_df = pd.DataFrame()
        jsons = []

        num = 1
        for file in dirs:
            if re.search(pattern, file):
                temp = pd.read_json('./Data/' + file, lines=True)
                print('Shape of %s is' % file, temp.shape)
                temp = self.add_label(temp)
                temp.to_csv('./Data/label' + str(num) + '_nov18.csv')
                num += 1
        print('Done!')

    def add_label(self, INPUT_PATH, OUTPUT_PATH: str):
        print('>>> Start to add corresponding label:')

        # Transform json into pandas dataframe
        self.df = pd.read_json(INPUT_PATH, lines=True)

        # Include tweets with only English as language
        df_en = self.df[['created_at', 'id', 'text']][self.df['lang'] == 'en']

        # Remove duplicates tweets
        tweets = df_en['text'].drop_duplicates()

        data = df_en.loc[tweets.index, :]

        # Get emoji mapping list
        mapping = [l.strip().split()
                   for l in open('mapping.txt', encoding='utf8')]
        map_dict = {}
        for m in mapping:
            map_dict[m[1]] = int(m[0])

        # Labeling tweets with corresponding emoji number
        emojis = unicode_codes.EMOJI_UNICODE
        emojis = sorted(unicode_codes.EMOJI_UNICODE.values(),
                        key=len, reverse=True)
        pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
        e = re.compile(pattern)

        def emoji_list(text):
            _ele = []

            def replace(match):
                loc = match.span()
                code = match.group(0)
                name = unicode_codes.UNICODE_EMOJI.get(code, None)
                if name:
                    _ele.append(
                        {"location": loc, "coding": code, "description": name})
                return code
            e.sub(replace, text)
            return _ele

        label_dict = {}
        for k, v in tweets.items():
            e_list = emoji_list(v)
            e_set = set([dic['coding'] for dic in e_list if 'coding' in dic])
            if len(e_set) == 1:
                emoji = e_set.pop()
                if emoji in map_dict:
                    label_dict[k] = map_dict[emoji]

        label_df = pd.DataFrame.from_dict(
            label_dict, orient='index', columns=['label'])

        # Join dataframe
        df_ = data.join(label_df, how='inner')
        print("File Shape:", df_.shape)

        df_.to_csv(OUTPUT_PATH)
        print('Done!')


if __name__ == '__main__':

    ct = 0
    label = labeling_tweets()

    for file in range(10):
        print(">>>.labeling file: ", file)
        INPUT_PATH = './Data/' + str(file) + 'RAWDATA_NOV17_ALLEMOJIS.json'
        OUTPUT_PATH = './Data/Labelled/_Labelled' + str(file) + 'DATA.csv'
        label.add_label(INPUT_PATH, OUTPUT_PATH)
        ct += 1
        print('Done!')

