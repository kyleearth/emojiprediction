from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import twitter_credentials


class TwitterStreamer():

    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename)
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,
                            twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                              twitter_credentials.ACCESS_TOKEN_SECRET)
        stream = Stream(auth, listener)

        stream.filter(track=hash_tag_list)


class StdOutListener(StreamListener):

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename
        self.num = 1

    def on_data(self, data):
        try:
            # print(data)
            print('Number of grab: %d' % self.num)
            self.num += 1
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    # INITIAL HERE
    FILENAME = './Data/raw_data_nov_16.json'
    HASH_TAG = ['🐶', '🙈', '♻️', '😡', '😋', '😱', '🙏', '👍', '👫',
                '🇺🇸', '😂', '🙃', '😘', '❤️', '🔥', '🌚', '💯', '🙌', '🔞', '😭']

    hash_tag_list = HASH_TAG
    fetched_tweets_filename = FILENAME

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
