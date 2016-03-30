from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import input_data
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_serving.session_bundle import exporter
import cifar10
import math
import sys


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def train():
  """Train CIFAR-10 for a number of steps."""

  #Load Dataset and related attributes
  dataset = input_data.read(FLAGS.input_dir)
  image, label = dataset.train_dataset
  image_size  = dataset.image_size
  paired_training_samples = nCr(dataset.train_samples,2) # Model will be trained on paired data



  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    train_images = tf.placeholder(tf.float32, shape=(2, FLAGS.batch_size, image_size[0], image_size[1], image_size[2]))
    labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size))

    eval_images = tf.placeholder(tf.float32, shape=(2, 1, image_size[0], image_size[1], image_size[2]))


    images, images_p = tf.split(0, 2, train_images)
    tf.image_summary('images', train_images)
    tf.image_summary('images_p', train_images)

    # For inference.cc
    eval_image, eval_image_p = tf.split(0, 2, eval_images)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope('inference') as scope:
      logits = cifar10.inference(tf.squeeze(images, [0]))
      scope.reuse_variables()
      logits2 = cifar10.inference(tf.squeeze(images_p, [0]))
      #evaluation_only
      eval_logits = cifar10.inference(tf.squeeze(eval_image, [0]))
      eval_logits2 = cifar10.inference(tf.squeeze(eval_image_p, [0]))


    # Calculate loss.
    loss = cifar10.loss(logits, logits2, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, FLAGS.max_steps, paired_training_samples)
    accuracy = cifar10.accuracy(logits, logits2, labels)

    eval_prediction = cifar10.predict(eval_logits, eval_logits2)
    

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), sharded=True)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
    i = 0
    j = 1
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      image_batch = [[],[]]
      label_batch = []
      for k in range(FLAGS.batch_size):
        image_batch[0].append(image[i])
        image_batch[1].append(image[j])

        if label[i]==label[j]:
          label_batch.append(0.0)
        else:
          label_batch.append(1.0)
        
        j = j + 1
        if j == dataset.train_samples:
          i = i + 1
          j = i + 1
        if i == dataset.train_samples:
          i = 0
          j = 1
      feed_dict = {train_images: image_batch, labels: label_batch}
      

      _, loss_value= sess.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      print(loss_value)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        model_exporter = exporter.Exporter(saver)
        signature = exporter.classification_signature(input_tensor=eval_images, scores_tensor=eval_prediction)
        model_exporter.init(sess.graph.as_graph_def(),
                      default_graph_signature=signature)
        model_exporter.export(FLAGS.export_path, 1, sess)

        saver.save(sess, checkpoint_path, global_step=step)
        print('Done exporting!')

def main(argv=None): 
 # pylint: disable=unused-argument
  if(len(argv)==6):
    tf.app.flags.DEFINE_string('input_dir', str(sys.argv[1]),
                           """Path to the data directory.""")
    tf.app.flags.DEFINE_integer('batch_size', int(sys.argv[2]),
                            """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_integer('max_steps', int(sys.argv[3]),
                            """Number of batches to run.""")
    tf.app.flags.DEFINE_string('train_dir', str(sys.argv[4]),
                           """Directory where to write event logs """
                           """and checkpoint.""")
    tf.app.flags.DEFINE_string('export_path', str(sys.argv[5]),
                       """Directory where to write event logs """
                       """and checkpoint.""")
    train()
  else:
    print("enter propoer arguments")

    



if __name__ == '__main__':
  tf.app.run()
