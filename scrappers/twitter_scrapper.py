from credentials import consumer_key, consumer_secret, access_token, \
    access_token_secret
import tweepy


class Twitter:

    def __init__(self):
        self.api = self.authenticate()

    @staticmethod
    def authenticate():
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return tweepy.API(auth, wait_on_rate_limit=True)

    def check_credentials(self):
        print("Credentials OK") if self.api.verify_credentials() else print(
            "Credentials not OK")

    def get_tweets(self, queried_string, lang="en", count=10):
        oldest_tweet_id = False
        filename = "output_{}.txt".format(
            queried_string.strip("#").split(" ")[0])
        with open(filename, "a") as f:
            for _ in range(count):
                for value in tweepy.Cursor(self.api.search, q=queried_string,
                                           lang=lang, tweet_mode='extended',
                                           max_id=oldest_tweet_id).items(200):
                    try:
                        cur_tweet_id = value.id
                        if not oldest_tweet_id or oldest_tweet_id > cur_tweet_id:
                            oldest_tweet_id = cur_tweet_id - 1
                        f.write(
                            value.retweeted_status.full_text.replace('\n', ''))
                        f.write('\n')
                    except AttributeError:
                        continue


a = Twitter()
# a.check_credentials()
a.get_tweets(queried_string="disgust filter:retweets", count=12)
a.get_tweets(queried_string="awful filter:retweets", count=12)
a.get_tweets(queried_string="unpleasant filter:retweets", count=12)
# a.get_tweets(queried_string="surprised -\"not surprised\" filter:retweets", count=30)
# a.get_tweets(queried_string="#surprise filter:retweets", count=30)
# a.get_tweets(queried_string="angry -\"not angry\" filter:retweets", count=30)
# a.get_tweets(queried_string="#anger filter:retweets", count=30)
# a.get_tweets(queried_string="#angry -\"angry bird\" -#birds -#Geronimo -#TokyoRevengers filter:retweets", count=30)