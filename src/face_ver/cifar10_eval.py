# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys

import numpy as np
import tensorflow as tf
import input_data
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

def eval_once(saver, summary_writer, top_k_op, summary_op, dataset, eval_images, labels):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  image, label = dataset.test_dataset
  num_examples = dataset.test_samples
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
      true_count = np.zeros([2,2])  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      i=0
      j=1
      while step < num_iter and not coord.should_stop():
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
        feed_dict = {eval_images: image_batch, labels: label_batch}
    
        predictions, summary_var = sess.run([top_k_op, summary_op], feed_dict=feed_dict)
        true_count += predictions
        print(true_count)
        step += 1

      # Compute precision @ 1.
      precision = (true_count[1,0]+true_count[0,1]) / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(summary_var)
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  dataset = input_data.read(FLAGS.input_dir)
  image_size = dataset.image_size
  with tf.Graph().as_default():
    # Build a Graph that computes the logits predictions from the
    # inference model.
    eval_images = tf.placeholder(tf.float32, shape=(2, FLAGS.batch_size, image_size[0], image_size[1], image_size[2]))
    labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size))

    images, images_p = tf.split(0, 2, train_images)
     
    with tf.variable_scope('inference') as scope:
      logits = cifar10.inference(images)
      scope.reuse_variables()
      logits2 = cifar10.inference(images_p)

    # Calculate predictions.
    accuracy = cifar10.accuracy(logits, logits2, labels)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      eval_once(saver, summary_writer, accuracy, summary_op, dataset, eval_images, labels)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
 if(len(argv)==6):
    tf.app.flags.DEFINE_string('input_dir', str(sys.argv[1]),
                           """Path to the data directory.""")
    tf.app.flags.DEFINE_integer('batch_size', int(sys.argv[2]),
                            """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_boolean('run_once', 1==int(sys.argv[3]),
                         """Whether to run eval only once.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', str(sys.argv[4]),
                           """Directory where to read model checkpoints.""")
    tf.app.flags.DEFINE_string('eval_dir', str(sys.argv[5]),
                           """Directory where to write event logs.""")

    evaluate()

  else:
    print("enter propoer arguments")


if __name__ == '__main__':
  tf.app.run()
