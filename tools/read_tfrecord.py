import os 
import matplotlib.pyplot as plt
import tensorflow as tf 
import argparse

BATCH_SIZE = 64
IMAGE_SIZE = [128,128]
AUTO = tf.data.experimental.AUTOTUNE

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
  image = decode_image(parsed_example['image'])
  return image

def decode_image(image_data):
    #decode image 
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,IMAGE_SIZE)
    return image

 #function to plot images
def view_image(ds):
    images = next(iter(ds)) # extract 1 batch from the dataset
    images = images.numpy()

    fig = plt.figure(figsize=(20, 20))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])

def get_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=AUTO)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def run_cmdLine(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Show image from TFRecordDataset')
    parser.add_argument('-r', '--tfrecords',    help='input file as tfrecord', dest='filename', action='append_const', const='tfrecords')

    args = parser.parse_args()
    if not args.filename:
        print('No tasks specified. Please see "-h" for help.')
        exit(1)
    ds = get_dataset(args.filename)
    view_image(ds)
    
if __name__ == "__main__":
    run_cmdline(sys.argv)


