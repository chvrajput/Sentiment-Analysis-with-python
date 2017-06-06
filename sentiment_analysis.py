# Python script
import nltk

pos_tweets = [('I love this car', 'positive'),
	('This view is amazing', 'positive'),
	('I feel great this morning', 'positive'),
	('I am so excited about the concert', 'positive'),
	('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
	('This view is horrible', 'negative'),
	('I feel tired this morning', 'negative'),
	('I am not looking forward to the concert', 'negative'),
	('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
	words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((words_filtered, sentiment))

test_tweets = [
	(['feel', 'happy', 'this', 'morning'], 'positive'),
	(['larry', 'friend'], 'positive'),
	(['not', 'like', 'that', 'man'], 'negative'),
	(['house', 'not', 'great'], 'negative'),
	(['your', 'song', 'annoying'], 'negative')]
# get the word lists of tweets
def get_words_in_tweets(tweets):
	all_words = []
	for (words, sentiment) in tweets:
		all_words.extend(words)
	return all_words

# get the unique word from the word list	
def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features = wordlist.keys()
	return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features
training_set = nltk.classify.util.apply_features(extract_features, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)

def train(labeled_featuresets, estimator=nltk.probability.ELEProbDist):
    # Create the P(label) distribution
    label_probdist = estimator(label_freqdist)
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    return NaiveBayesClassifier(label_probdist, feature_probdist)

tweet_positive = 'Larry is my friend'
tweet_negative = 'Larry is not my friend'

print classifier.classify(extract_features(tweet_positive.split()))
# > positive
print classifier.classify(extract_features(tweet_negative.split()))
# > negative

tweet_negative2 = 'Your song is annoying'
print classifier.classify(extract_features(tweet_negative2.split()))

def classify_tweet(tweet):
    return \
        classifier.classify(extract_features(tweet)) # nltk.word_tokenize(tweet)

total = accuracy = float(len(test_tweets))

for tweet in test_tweets:
    if classify_tweet(tweet[0]) != tweet[1]:
        accuracy -= 1

print('Total accuracy: %f%% (%d/20).' % (accuracy / total * 100, accuracy))
