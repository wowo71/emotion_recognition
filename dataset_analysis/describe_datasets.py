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

df1 = pd.read_csv("../dataset/internet_dataset/isear/isear_final_no_shame_and_guilt.csv", sep='@')
print(df1.count())
# print(df.describe())
df2 = pd.read_csv("../dataset/internet_dataset/tweet_emotions/tweet_emotions_v4_basic_emotions.csv", sep=',')
print(df2.count())
df3 = pd.read_csv("/home/wojtek/Desktop/emotion_recognition/dataset/my_dataset/filtered_outputs/output_obrzydzenie_filtered_cleaned.txt", sep=',')
print(df3.count())
# df3 = pd.read_csv("../dataset/my_dataset/tweet_emotions/tweet_emotions_v4_basic_emotions.csv", sep=',')
# df = pd.concat([df1, df2, df3])
# print(df.count())

