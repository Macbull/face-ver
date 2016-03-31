# Copyright 2016 Google Inc. All Rights Reserved.
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

#!/usr/grte/v4/bin/python2.7

"""A client that talks to mnist_inference service.

The client downloads test images of mnist data set, queries the service with
such test images to get classification, and calculates the inference error rate.
Please see mnist_inference.proto for details.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf
from scipy import ndimage
import numpy as np
from tensorflow_serving.example import inference_pb2

tf.app.flags.DEFINE_string('image1_file', 'test/img1.pgm', 'Input image 1')
tf.app.flags.DEFINE_string('image2_file', 'test/img1.pgm', 'Input image 2')
tf.app.flags.DEFINE_integer('pixel_depth', 255,
                            'maximum intensity of pixel')
tf.app.flags.DEFINE_string('server', '', 'mnist_inference service host:port')

FLAGS = tf.app.flags.FLAGS

image_size = [112,92]

def do_inference(hostport, input):
  """Tests mnist_inference service with concurrent requests.

  Args:
    hostport: Host:port address of the mnist_inference service.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = mnist_inference_pb2.beta_create_MnistService_stub(channel)
  result = {'active': 0, 'error': 0, 'done': 0}
  request = inference_pb2.VerifyRequest()
  for pixel in data.flatten():
    request.image_data.append(pixel.item())
  result_future = stub.Classify(request, 5.0)
  exception = result_future.exception()
  if exception:
    print exception
  else:
    sys.stdout.write('.')
    sys.stdout.flush()
    response = numpy.array(result_future.message)
    print(response)




def main(_):
  if FLAGS.num_tests > 10000:
    print 'num_tests should not be greater than 10k'
    return
  if not FLAGS.server:
    print 'please specify server host:port'
    return

  image1 = (ndimage.imread(image1_file).astype(float)*2 - pixel_depth) /pixel_depth
  image2 = (ndimage.imread(image2_file).astype(float)*2 - pixel_depth) /pixel_depth
  data = np.ndarray(shape = (2, 1, image_size[0], image_size[1],1), dtype=np.float32)
  data[0,0,:,:,0] = image1
  data[1,0,:,:,0] = image2

  result = do_inference(FLAGS.server, data,
                            FLAGS.concurrency, FLAGS.num_tests)


  


if __name__ == '__main__':
  tf.app.run()
