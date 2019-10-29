import tensorflow as tf
import numpy as np
import pandas as pd


def csv2tfrecords(input_path, output_path):
    data_frame = pd.read_csv(input_path)
    print(data_frame.head())
    data_values = data_frame.values
    print("values shape: ", data_values.shape)

    writer = tf.python_io.TFRecordWriter(output_path)

    for i in range(data_values.shape[0]):
        data_raw = data_values[i]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_raw[0]])),
                    "start_time": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw[1].encode()])),
                    "end_time": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw[2].encode()])),
                    "duration": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_raw[3]])),
                    "start_soc": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_raw[4]])),
                    "end_soc": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_raw[5]])),
                    "state": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw[6].encode()])),
                    "max_temp": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_raw[7]])),
                    "min_temp": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_raw[8]])),
                    "max_current": tf.train.Feature(float_list=tf.train.FloatList(value=[data_raw[9]])),
                    "min_current": tf.train.Feature(float_list=tf.train.FloatList(value=[data_raw[10]])),
                    "max_voltage": tf.train.Feature(float_list=tf.train.FloatList(value=[data_raw[11]])),
                    "min_voltage": tf.train.Feature(float_list=tf.train.FloatList(value=[data_raw[12]]))
                }
            )
        )
        writer.write(record=example.SerializeToString())

    writer.close()


def getRecords(input_path):
    with tf.Session() as sess:
        example = tf.train.Example()

        # train_record表示训练的tfrecords文件的路径
        record_iterator = tf.python_io.tf_record_iterator(path=input_path)
        for record in record_iterator:
            example.ParseFromString(record)
            f = example.features.feature
            id = f['id'].int64_list.value[0]
            start_time = f['start_time'].bytes_list.value[0]
            end_time = f['end_time'].bytes_list.value[0]
            min_voltage = f['min_voltage'].float_list.value[0]
            print(id, start_time, end_time, min_voltage)


if __name__ == "__main__":
    input_path = "/Users/alanp/Projects/AbleCloud/data/10001737_output.csv"
    output_path = "/Users/alanp/Projects/AbleCloud/data/10001737.tfrecords"
    # csv2tfrecords(input_path, output_path)
    getRecords(output_path)
