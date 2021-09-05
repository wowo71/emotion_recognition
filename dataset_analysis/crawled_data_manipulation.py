from os import walk

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


remove_duplicates_from_files_from_dir(summarized_outputs)
