import pandas as pd

# df = pd.read_csv("../dataset/internet_dataset/isear.csv", sep='|')
# df = df[["Field1", "SIT"]]
# df = df.rename(columns={"Field1": "emotion", "SIT": "comment"})
# df.to_csv("../dataset/internet_dataset/isear_removed_columns.csv", sep="@", index=False)

# df = pd.read_csv("../dataset/internet_dataset/isear_removed_columns.csv", sep='@')
# print(df.count())
# print(df.describe())
# df = df.drop(df[df.comment.str.startswith("[") & df.comment.str.endswith("]")].index)
# df = df.drop(df[df.comment == "NO RESPONSE."].index)
# df = df.drop(df[df.comment == "Doesn't apply."].index)
# df = df.drop(df[df.comment == "Friends who torture animals."].index)
# print(df.count())
# print(df.describe())
# df = df.drop_duplicates(subset="comment")
# print(df.count())
# print(df.describe())
# print(df['emotion'].value_counts())
# df.to_csv("../dataset/internet_dataset/isear_cleared.csv", sep="@", index=False)

# df = pd.read_csv("../dataset/internet_dataset/isear_cleared.csv", sep='@')
#
# from spellchecker import SpellChecker
#
# spell = SpellChecker()
#
# def spell_check(x):
#     correct_word = []
#     mispelled_word = x.split()
#     for word in mispelled_word:
#         correct_word.append(spell.correction(word))
#     print("dziala")
#     return ' '.join(correct_word)
#
#
# df['comment'] = df['comment'].apply(lambda x: spell_check(x))

# df.to_csv("../dataset/internet_dataset/isear_cleared_spellchecked.csv", sep="@", index=False)

# df = pd.read_csv("../dataset/internet_dataset/isear_cleared_spellchecked.csv", sep='@')
# df = df.drop(df[df.emotion == "shame"].index)
# df = df.drop(df[df.emotion == "guilt"].index)
# df.to_csv("../dataset/internet_dataset/isear_no_shame_and_guilt.csv", sep="@", index=False)


# df = pd.read_csv("../dataset/internet_dataset/tweet_emotions.csv", sep=',')
# print(df.head())
# df = df[["sentiment", "content"]]
# df = df.rename(columns={"sentiment": "emotion", "content": "comment"})
# df.to_csv("../dataset/internet_dataset/tweet_emotions_removed_columns.csv", sep=",", index=False)

# import re
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/internet_dataset/tweet_emotions_v2_removed_tags_and_links.csv", "r") as f1:
#     text = f1.read()
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/internet_dataset/tweet_emotions_v2_removed_tags_and_links.csv", "w") as f2:
#     text = re.sub(
#         r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
#         '', text, flags=re.MULTILINE)
#     text = (re.sub(r"[@](.*?)[\s]", '', text, flags=re.MULTILINE))
#     f2.write(text)

# df = pd.read_csv("../dataset/internet_dataset/tweet_emotions_v2_removed_tags_and_links.csv", sep=',', error_bad_lines=False)
# df = df.drop_duplicates(subset="comment")
# print(df.count())
# print(df.describe())
# df.to_csv("../dataset/internet_dataset/tweet_emotions_v3_no_duplicates.csv", sep=",", index=False)

# df = pd.read_csv("../dataset/internet_dataset/tweet_emotions_v3_no_duplicates.csv", sep=',', error_bad_lines=False)
# df = df.drop(df[df.emotion == "neutral"].index)
# df = df.drop(df[df.emotion == "hate"].index)
# df = df.drop(df[df.emotion == "empty"].index)
# df = df.drop(df[df.emotion == "boredom"].index)
# df = df.drop(df[df.emotion == "enthusiasm"].index)
# df = df.drop(df[df.emotion == "relief"].index)
# df = df.drop(df[df.emotion == "fun"].index)
# df = df.drop(df[df.emotion == "love"].index)
# print(df['emotion'].value_counts())
# print(df.count())
# print(df.describe())
# df.to_csv("../dataset/internet_dataset/tweet_emotions_v4_basic_emotions.csv", sep=",", index=False)


#### CREATE FINAL DATASET

# df1 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset.csv", sep=',')
# print(df1.count())
# print(df1.describe())
# df2 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/scrappers/4/output_angry_filtered_cleaned_filtered.txt", sep=',')
# print(df2.count())
# df3 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/scrappers/4/output_disgusted_filtered_cleaned_filtered_filtered.txt", sep=',')
# print(df3.count())
# df4 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/scrappers/4/output_surprised_filtered_cleaned_filtered.txt", sep=',')
# print(df4.count())
# # df5 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/my_dataset/filtered_outputs/output_strach_filtered_cleaned.txt", sep=',')
# # print(df5.count())
# # df6 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/my_dataset/filtered_outputs/output_szczescie_filtered_cleaned.txt", sep=',')
# # print(df6.count())
# # df7 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/my_dataset/filtered_outputs/output_zaskoczenie_filtered_cleaned.txt", sep=',')
# # print(df7.count())
# # df8 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/my_dataset/filtered_outputs/output_zlosc_filtered_cleaned.txt", sep=',')
# # print(df8.count())
# df = pd.concat([df1, df2, df3, df4])
# # print(df.count())
# print(df.count())
# print(df.describe())
# print(df['emotion'].value_counts())
# df.to_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v2.csv", sep=",", index=False)


#### ANALYSE DATA ####
# df = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v2.csv", sep=',')
# df = df.drop_duplicates(subset="comment")
# import re
#


# ---- rozwijam skroty i usuwam stopwordy
# import re
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v5.csv", "r") as f1:
#     text = f1.read()
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v5_no_stopwords.csv", "a") as f2:
#     text = (re.sub("n't", ' not', text, flags=re.MULTILINE))
#     f2.write(text)
# import nltk
# from nltk.corpus import stopwords
# # nltk.download('stopwords')
# # nltk.download('punkt')
# from nltk.tokenize import word_tokenize
#
# stopwords = stopwords.words('english')
# stopwords.remove('nor')
# stopwords.remove('not')
# stopwords.remove('no')
# stopwords.append('ca')
# stopwords.append(',')
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v5.csv", "r") as f1:
#     text = f1.readlines()
# for line in text:
#     emotion, comment = line.split(",", 1)
#     text_tokens = word_tokenize(comment)
#     with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v5_no_stopwords.csv", "a") as f2:
#         text = ' '.join([word for word in text_tokens if not word in stopwords])
#         f2.write(emotion + "," + text + "\n")


### usuwanie pozostalosci po stopwordach
# import re
#
# with open(
#         "/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_clear.csv",
#         "r") as f1:
#     text = f1.read()
# with open(
#         "/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_clear_round2.csv",
#         "a") as f2:
#     text = (re.sub("'m", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("'re", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("'", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("'", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("#", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("_", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\*", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\.", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\-", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\?", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\(", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\[", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\]", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\)", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\!", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\`", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\;", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\:", ' ', text, flags=re.MULTILINE))
#     text = (re.sub(r"\:", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("& amp", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("& quot", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("& lt", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("& gt", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     text = (re.sub("  ", ' ', text, flags=re.MULTILINE))
#     f2.write(text)
# from nltk.tokenize import word_tokenize
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_clear_round2.csv", "r") as f1:
#     text = f1.readlines()
# for line in text:
#     emotion, comment = line.split(",", 1)
#     text_tokens = word_tokenize(comment)
#     with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_clear_round3.csv", "a") as f2:
#         if len(text_tokens) > 1:
#             f2.write(emotion + "," + ' '.join(text_tokens) + "\n")

### removing non utf-8 comments
# import re
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_clear_round3_test.csv", "r") as f1:
#     text = f1.read()
#     pattern = re.compile('(?i)[^a-z0-9, \n]+')
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_utf_8.csv", "a") as f2:
#     f2.write(pattern.sub('', text))

#### ANALYSE DATA ####
# df = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v6_utf_8.csv", sep=',', error_bad_lines=False)
# print(df.count())
# df = df.drop_duplicates(subset="comment")
# print(df.count())
# print(df['emotion'].value_counts())
# df.to_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v7.csv", sep=",", index=False)


#### LEMMA

# import nltk
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# nltk.download('averaged_perceptron_tagger')
# from nltk.corpus import wordnet
# from nltk.tokenize import word_tokenize
#
# def pos_tagger(nltk_tag):
#     if nltk_tag.startswith('J'):
#         return wordnet.ADJ
#     elif nltk_tag.startswith('V'):
#         return wordnet.VERB
#     elif nltk_tag.startswith('N'):
#         return wordnet.NOUN
#     elif nltk_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return None
#
# lemmatizer = WordNetLemmatizer()
# with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v7_lemmatized.csv", "r") as f1:
#     text = f1.readlines()
# for line in text:
#     emotion, comment = line.split(",", 1)
#     pos_tagged = nltk.pos_tag(nltk.word_tokenize(comment))
#     wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
#     lemmatized_sentence = []
#     for word, tag in wordnet_tagged:
#         if tag is None:
#             # if there is no available tag, append the token as is
#             lemmatized_sentence.append(word)
#         else:
#             # else use the tag to lemmatize the token
#             lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
#     lemmatized_sentence = " ".join(lemmatized_sentence)
#     with open("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v7_lemmatized_pos.csv", "a") as f2:
#         f2.write(emotion + "," + lemmatized_sentence + "\n")

#### ANALYSE DATA ####
df = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v7_lemmatized_pos.csv", sep=',', error_bad_lines=False)
print(df.count())
df = df.drop_duplicates(subset="comment")
print(df.count())
print(df['emotion'].value_counts())
df = df.groupby('emotion').head(3000)
print(df['emotion'].value_counts())
df.to_csv("/home/wojtek/Desktop/emotion_recognition/dataset/final_dataset_v8_top_3000.csv", sep=",", index=False)
