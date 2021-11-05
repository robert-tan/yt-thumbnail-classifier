import json
import matplotlib.pyplot as plt
import os
import re
import requests
import tensorflow as tf
from urllib.request import urlopen


def read_categories():
    with open("categories.json") as f:
        categories = json.load(f)
    return categories

CATEGORIES = read_categories()


def get_file_paths():
    files = [f for f in os.listdir(".") if re.match(r"train[0-9]{4}\.tfrecord", f)]
    processed_files = [f"{f.split('.')[0]}_processed.tfrecord" for f in files]
    filtered_files = [(f, processed_files[i]) for i, f in enumerate(files) if not os.path.isfile(processed_files[i])]
    return filtered_files


def read_dataset(file_path):
    dataset = tf.data.TFRecordDataset(file_path)

    def read_tfrecord(serialized_example):
        feature_description = {
            'labels': tf.io.VarLenFeature(tf.int64),
            'id': tf.io.VarLenFeature(tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        return example["id"].values, example["labels"].values

    return dataset.map(read_tfrecord)


def write_dataset(file_path, dataset):
    def serialize_example(thumbnail, labels):
        n_W, n_H, n_C = thumbnail.shape
        thumbnail_bytes = thumbnail.flatten().tobytes()
        feature = {
            'thumbnail': tf.train.Feature(bytes_list=tf.train.BytesList(value=[thumbnail_bytes])),
            'n_W': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_W])),
            'n_H': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_H])),
            'n_C': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_C])),
            'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(file_path) as writer:
        for thumbnail, labels in dataset:
            serialized_example = serialize_example(thumbnail, labels)
            writer.write(serialized_example)


def process_dataset(dataset):
    def ytID_to_ndarray(yt_id):
        image_url = "https://img.youtube.com/vi/" + yt_id + "/0.jpg"
        try:
            f = urlopen(image_url)
            a = plt.imread(f, 0)
            return a
        except:
            # print(f"Cannot download thumbnail for {yt_id}")
            return None


    def video_id(vid):
        url = f"http://data.yt8m.org/2/j/i/{vid[:2]}/{vid}.js"
        r = requests.get(url)
        if r.status_code != 200:
            # print(f"{vid} returned status code {r.status_code}")
            return None
        return r.text[10:-3]


    def convert_labels(labels):
        labels = labels.numpy()
        return [CATEGORIES[str(label)] for label in labels]

    res = list()
    fails = 0
    for i, (vid, labels) in enumerate(dataset):
        id_str = vid[0].numpy().decode("utf-8")
        yt_id = video_id(id_str)
        if yt_id is None:
            fails += 1
            continue

        thumbnail = ytID_to_ndarray(yt_id)
        if thumbnail is None:
            fails += 1
            continue

        res.append((thumbnail, convert_labels(labels)))
        if i % 100 == 0:
            print(f"Processing examples {i}-{i + 99}, last 100 had {fails} failures")
            fails = 0
    return res


if __name__ == "__main__":
    file_paths = get_file_paths()
    for file_path, processed_file_path in file_paths:
        print(f"Processing {file_path}")
        dataset = read_dataset(file_path)
        processed = process_dataset(dataset)
        write_dataset(processed_file_path, processed)
        print(f"Wrote {processed_file_path}")

