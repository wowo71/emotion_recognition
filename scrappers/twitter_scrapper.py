from credentials import consumer_key, consumer_secret, access_token, access_token_secret
import tweepy


class Twitter:

    def __init__(self):
        self.api = self.authenticate()

    @staticmethod
    def authenticate():
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return tweepy.API(auth)

    def check_credentials(self):
        print("Credentials OK") if self.api.verify_credentials() else print("Credentials not OK")

    def get_tweets(self, queried_string, lang="en", count=5):
        oldest_tweet_id = False
        filename = "output_{}.txt".format(queried_string.strip("#").split(" ")[0])
        with open(filename, "a") as f:
            for _ in range(count):
                for value in tweepy.Cursor(self.api.search, q=queried_string, lang=lang, tweet_mode='extended', max_id=oldest_tweet_id).items(200):
                    # print(value.id)
                    try:
                        current_tweet_id = value.id
                        if not oldest_tweet_id or oldest_tweet_id > current_tweet_id:
                            oldest_tweet_id = current_tweet_id - 1
                        f.write(value.retweeted_status.full_text.replace('\n', ''))
                        f.write('\n')
                    except AttributeError:
                        continue

a = Twitter()
# a.check_credentials()
a.get_tweets(queried_string="#happy AND -birthday filter:retweets", count=10)
