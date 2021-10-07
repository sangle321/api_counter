# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for reading and updating configuration files."""


import os

from google.protobuf import text_format
import tensorflow as tf

from tensorflow.python.lib.io import file_io

def get_image_resizer_config(model_config):
  """Returns the image resizer config from a model config.

  Args:
    model_config: A model_pb2.DetectionModel.

  Returns:
    An image_resizer_pb2.ImageResizer.

  Raises:
    ValueError: If the model type is not recognized.
  """
  meta_architecture = model_config.WhichOneof("model")
  meta_architecture_config = getattr(model_config, meta_architecture)

  if hasattr(meta_architecture_config, "image_resizer"):
    return getattr(meta_architecture_config, "image_resizer")
  else:
    raise ValueError("{} has no image_reszier_config".format(
        meta_architecture))


def get_spatial_image_size(image_resizer_config):
  """Returns expected spatial size of the output image from a given config.

  Args:
    image_resizer_config: An image_resizer_pb2.ImageResizer.

  Returns:
    A list of two integers of the form [height, width]. `height` and `width` are
    set  -1 if they cannot be determined during graph construction.

  Raises:
    ValueError: If the model type is not recognized.
  """
  if image_resizer_config.HasField("fixed_shape_resizer"):
    return [
        image_resizer_config.fixed_shape_resizer.height,
        image_resizer_config.fixed_shape_resizer.width
    ]
  if image_resizer_config.HasField("keep_aspect_ratio_resizer"):
    if image_resizer_config.keep_aspect_ratio_resizer.pad_to_max_dimension:
      return [image_resizer_config.keep_aspect_ratio_resizer.max_dimension] * 2
    else:
      return [-1, -1]
  if image_resizer_config.HasField(
      "identity_resizer") or image_resizer_config.HasField(
          "conditional_shape_resizer"):
    return [-1, -1]
  raise ValueError("Unknown image resizer type.")


def get_max_num_context_features(model_config):
  """Returns maximum number of context features from a given config.

  Args:
    model_config: A model config file.

  Returns:
    An integer specifying the max number of context features if the model
      config contains context_config, None otherwise

  """
  meta_architecture = model_config.WhichOneof("model")
  meta_architecture_config = getattr(model_config, meta_architecture)

  if hasattr(meta_architecture_config, "context_config"):
    return meta_architecture_config.context_config.max_num_context_features


def get_context_feature_length(model_config):
  """Returns context feature length from a given config.

  Args:
    model_config: A model config file.

  Returns:
    An integer specifying the fixed length of each feature in context_features.
  """
  meta_architecture = model_config.WhichOneof("model")
  meta_architecture_config = getattr(model_config, meta_architecture)

  if hasattr(meta_architecture_config, "context_config"):
    return meta_architecture_config.context_config.context_feature_length


def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
  """Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override pipeline_config_path.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  if config_override:
    text_format.Merge(config_override, pipeline_config)
  return create_configs_from_pipeline_proto(pipeline_config)



def create_configs_from_pipeline_proto(pipeline_config):
  """Creates a configs dictionary from pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config: pipeline_pb2.TrainEvalPipelineConfig proto object.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_configs`. Value are
      the corresponding config objects or list of config objects (only for
      eval_input_configs).
  """
  configs = {}
  configs["model"] = pipeline_config.model
  configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
  configs["eval_config"] = pipeline_config.eval_config
  configs["eval_input_configs"] = pipeline_config.eval_input_reader
  # Keeps eval_input_config only for backwards compatibility. All clients should
  # read eval_input_configs instead.
  if configs["eval_input_configs"]:
    configs["eval_input_config"] = configs["eval_input_configs"][0]
  if pipeline_config.HasField("graph_rewriter"):
    configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

  return configs

def save_pipeline_config(pipeline_config, directory):
  """Saves a pipeline config text file to disk.

  Args:
    pipeline_config: A pipeline_pb2.TrainEvalPipelineConfig.
    directory: The model directory into which the pipeline config file will be
      saved.
  """
  if not file_io.file_exists(directory):
    file_io.recursive_create_dir(directory)
  pipeline_config_path = os.path.join(directory, "pipeline.config")
  config_text = text_format.MessageToString(pipeline_config)
  with tf.gfile.Open(pipeline_config_path, "wb") as f:
    tf.logging.info("Writing pipeline config file to %s",
                    pipeline_config_path)
    f.write(config_text)
