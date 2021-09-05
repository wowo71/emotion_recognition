from os import walk
import re

summarized_outputs = "/home/wojtek/Desktop/emotion_recognition/dataset/sumarized_outputs"
filtered_outputs = "/home/wojtek/Desktop/emotion_recognition/dataset/filtered_outputs"


def remove_duplicates_from_files_from_dir(dataset_dir):
    filenames = next(walk(dataset_dir), (None, None, []))[2]
    for filename in filenames:
        file_abs_path = dataset_dir + "/" + filename
        with open(file_abs_path, "r") as f1:
            lines = set(f1.readlines())
        filtered_file_abs_path = filtered_outputs + "/" + filename.split(".")[
            0] + "_filtered.txt"
        with open(filtered_file_abs_path, "w") as f2:
            f2.writelines(lines)


def remove_links_and_tagged_accounts(filtered_output_dir):
    filenames = next(walk(filtered_output_dir), (None, None, []))[2]
    for filename in filenames:
        file_abs_path = filtered_output_dir + "/" + filename
        with open(file_abs_path, "r") as f1:
            text = f1.read()
        cleared_file_abs_path = filtered_outputs + "/" + filename.split(".")[
            0] + "_cleaned.txt"
        with open(cleared_file_abs_path, "w") as f2:
            text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', text, flags=re.MULTILINE)
            text = (re.sub(r"[@](.*?)[\s]", '', text, flags=re.MULTILINE))
            for _ in range(5):
                text = (re.sub(r"  ", ' ', text, flags=re.MULTILINE))
            f2.write(text)


# remove_duplicates_from_files_from_dir(summarized_outputs)
remove_links_and_tagged_accounts(filtered_outputs)
