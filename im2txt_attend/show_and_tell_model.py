# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from im2txt_attend.ops import image_embedding
from im2txt_attend.ops import image_processing
from im2txt_attend.ops import inputs as input_ops


class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.
    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
      encoded_image = None
      caption = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.encoded_images = encoded_image
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask
    
  def build_embedding_parameters(self):
    """ Build parameters for the image embeddings
    
    Outputs:
      self.image_embedding_weights
      self.image_embedding_biases
    """
    self.image_embedding_weights = tf.get_variable(
        "image_embedding_weights",
        shape=[2048, self.config.embedding_size],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    self.image_embedding_biases = tf.get_variable(
        "image_embedding_biases",
        shape=[self.config.embedding_size],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())
        
  def build_attention_parameters(self):
    """ Build parameters for the attention proposal network
    
    Outputs:
      self.image_attention_weights
      self.image_attention_weights_expand_x
      self.image_attention_weights_expand_y
      self.image_attention_biases
      self.image_attention_biases_expand_x
      self.image_attention_biases_expand_y
    """
    self.image_attention_weights = tf.get_variable(
        "image_attention_weights",
        shape=[2048 + (self.config.num_lstm_units*2), 1],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    self.image_attention_biases = tf.get_variable(
        "image_attention_biases",
        shape=[1, 1, 1, 1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())
    
  def build_context_tensor(self):
    """ Computes a context tensor from the inception output
    
    Inputs:
      self.attention_map
      self.inception_output
    
    Outputs:
      self.context_vector
      self.image_embeddings
    """

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.nn.relu(
        tf.tensordot(
          self.context_tensor, 
          self.image_embedding_weights, 1) + self.image_embedding_biases)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings
  
  def build_feature_tensor(self):
    """ Computes the inception tensor
    
    Inputs:
      self.images
    
    Outputs:
      self.inception_output
    """
    
    self.inception_output = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    
    # Brandon Trabucco 2018.06.13: attention is a 4 tensor of shape [batch_size, height, width, 1]
    self.build_attention_tensor(tf.zeros([
        tf.shape(self.inception_output)[0], 
        self.config.num_lstm_units*2], dtype=tf.float32))
    
    # Compute a prelimiary context tensor for the image
    self.build_context_tensor()

    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
    
  def build_attention_tensor(self, current_state):
    """ Computes a proposed attention tensor
    
    Inputs:
      current_state
      self.inception_output
    
    Outputs:
      self.context_tensor
    """
    
    # Brandon Trabucco 2018.06.14: concatenate the image features with lstm state
    state_tensor = tf.tile(
        tf.expand_dims(
            tf.expand_dims(current_state, axis=1), axis=2),
        [1, tf.shape(self.inception_output)[1], 
            tf.shape(self.inception_output)[2], 1])
    attend_input_tensor = tf.concat(
        [state_tensor, 
            tf.tile(self.inception_output, 
                    # because model runs beam search on parallel caption states
                    [tf.shape(state_tensor)[0] if self.mode == "inference" else 1, 
                     1, 1, 1])], 
        axis=3)
    
    # Run the attention proposal net
    unadjusted_attention = tf.tensordot(
      attend_input_tensor,
      self.image_attention_weights,
      1) + self.image_attention_biases
    
    # Normalize the attention using softmax
    adjusted_attention = tf.reshape(
      tf.nn.softmax(
        tf.reshape(unadjusted_attention,
          [tf.shape(unadjusted_attention)[0], -1])),
      tf.shape(unadjusted_attention))
    
    # Compute the attended feature tensor
    global_features = tf.reduce_sum(
        self.inception_output * adjusted_attention,
        axis=[1, 2])
    
    self.context_tensor = global_features
    
  def build_concat_embeddings(
      self,
      seq_embeddings=None, 
      singleton=False, 
      curry=False):
    """ Computes concatenated sequence and image embeddings
    
    Inputs:
      seq_embeddings
      self.image_embeddings
    
    Outputs:
      concat_image_seq_embeddings
    """
    
    self.build_context_tensor()
    if seq_embeddings is None:
        # Brandon Trabucco 2018.06.12: Concatenate image embedding with seq embedding
        # Brandon Trabucco 2018.06.13: Adding attention mechanism on image features
        return tf.concat([
            tf.zeros([tf.shape(self.image_embeddings)[0], self.config.embedding_size], dtype=tf.float32),
            self.image_embeddings
        ], 1) # this has shape [batch_size, embedding_size * 2]
        
    elif singleton:
        # Brandon Trabucco 2018.06.12: Concatenate image embedding with seq embedding
        # Brandon Trabucco 2018.06.13: Adding attention mechanism on image features
        return tf.concat([
            seq_embeddings,
            self.image_embeddings
        ], 1) # this has shape [batch_size, embedding_size * 2]
    
    elif curry:
        # Brandon Trabucco 2018.06.12: Concatenate image embedding with seq embedding
        # Brandon Trabucco 2018.06.13: Adding attention mechanism on image features
        return lambda: tf.concat([
            seq_embeddings(),
            self.image_embeddings
        ], 1) # this has shape [batch_size, embedding_size * 2]
        
    else:
        # Brandon Trabucco 2018.06.12: Concatenate image embedding with seq embedding
        # Brandon Trabucco 2018.06.13: Adding attention mechanism on image features
        return tf.concat([
            seq_embeddings,
            tf.tile(tf.expand_dims(self.image_embeddings, axis=1), [1, tf.shape(seq_embeddings)[1], 1])
        ], 2) # this has shape [batch_size, padded_length, embedding_size * 2]

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)
    
    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
        
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=tf.shape(self.image_embeddings)[0], dtype=tf.float32)
      _, initial_state = lstm_cell(
          # Brandon Trabucco 2018.06.13: Changed input to include image embedding
          #self.image_embeddings, 
          self.build_concat_embeddings(),
          zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
        
        # Compute an attention map of the image features
        self.build_attention_tensor(state_feed)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            # Brandon Trabucco 2018.06.13: Changed input to include image embedding
            #inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
            inputs=self.build_concat_embeddings(
                seq_embeddings=tf.squeeze(self.seq_embeddings, axis=[1]), 
                singleton=True),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
      else:
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        
        # Brandon Trabucco 2018.06.12: Define raw rnn to use for attention
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(self.seq_embeddings)[1])
        inputs_ta = inputs_ta.unstack(tf.transpose(self.seq_embeddings, [1, 0, 2]))
        
        def loop_fn(time, cell_output, cell_state, loop_state):
          next_state = (initial_state 
              if cell_output is None else cell_state)
          finished = (time >= sequence_length) 
          
          # Compute attention map of image features.
          self.build_attention_tensor(tf.concat(axis=1, values=next_state))
        
          next_input = tf.cond(
            tf.reduce_all(finished),
            lambda: tf.zeros([
                self.config.batch_size,  
                self.config.embedding_size*2], dtype=tf.float32),
            self.build_concat_embeddings(
                seq_embeddings=lambda: inputs_ta.read(time),
                curry=True))
            
          next_input = tf.reshape(next_input, [
              self.config.batch_size, 
              self.config.embedding_size*2])
        
          return (finished, next_input, next_state, cell_output, None)
    
        # Run the batch of sequence embeddings through the LSTM.
        outputs_ta, _final_state, _ = tf.nn.raw_rnn(lstm_cell,
                                             loop_fn,
                                             scope=lstm_scope)
        lstm_outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    self.unscaled_logits = logits
    self.softmax_outputs = tf.nn.softmax(logits, name="softmax")
    
    if self.mode != "inference":
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                              logits=logits)
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")
      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    # Brandon Trabucco 2018.06.13: Adding attention mechanism on image features
    self.build_embedding_parameters()
    self.build_attention_parameters()
    #self.build_image_embeddings()
    self.build_feature_tensor()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
