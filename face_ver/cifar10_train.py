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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  dataset = input_data.read()
  image, image_p, label = dataset.train_dataset
  image_size  = dataset.image_size
  batch_size = 128
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    train_images = tf.placeholder(tf.float32, shape=(2, batch_size, image_size[0], image_size[1], image_size[2]))
    eval_images = tf.placeholder(tf.float32, shape=(2, 1, image_size[0], image_size[1], image_size[2]))
    labels = tf.placeholder(tf.float32, shape=(batch_size))
    # tf.image_summary('images', train_images)

    images, images_p = tf.split(0, 2, train_images)
    eval_image, eval_image_p = tf.split(0, 2, train_images)
    # tf.image_summary('images2', images)
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
    train_op = cifar10.train(loss, global_step)

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

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      offset = (step * batch_size) % (dataset.train_samples - batch_size)
      # _, loss_value= sess.run([train_op, loss], feed_dict={images: image[offset:(offset + batch_size)], images2: image_p[offset:(offset + batch_size)], labels: 1.0*label[offset:(offset + batch_size)]})
      _, loss_value= sess.run([train_op, loss], feed_dict={train_images: [image[offset:(offset + batch_size)],image_p[offset:(offset + batch_size)]], labels: 1.0*label[offset:(offset + batch_size)]})

      # _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict={images: image[offset:(offset + batch_size)], images2: image_p[offset:(offset + batch_size)], labels: 1.0*label[offset:(offset + batch_size)]})
      
      duration = time.time() - start_time

      print(loss_value)
      # print(acc)
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
        summary_str = sess.run(summary_op, feed_dict={train_images: [image[offset:(offset + batch_size)],image_p[offset:(offset + batch_size)]], labels: 1.0*label[offset:(offset + batch_size)]} )
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        model_exporter = exporter.Exporter(saver)
        signature = exporter.classification_signature(input_tensor=eval_images, scores_tensor=eval_prediction)
        model_exporter.init(sess.graph.as_graph_def(),
                      default_graph_signature=signature)
        export_path = sys.argv[-1]
        model_exporter.export(export_path, 1, sess)

        saver.save(sess, checkpoint_path, global_step=step)
        print('Done exporting!')

def main(argv=None): 
 # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
