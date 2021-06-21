import os 
import matplotlib.pyplot as plt

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
  image = decode_image(parsed_example['image'])

  return image

def decode_image(image_data):
    #decode image 
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,[IMG_HEIGHT,IMG_WIDTH])
    return image

 #function to plot images
def view_image(ds):
    images = next(iter(ds)) # extract 1 batch from the dataset
    images = images.numpy()

    fig = plt.figure(figsize=(20, 20))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
