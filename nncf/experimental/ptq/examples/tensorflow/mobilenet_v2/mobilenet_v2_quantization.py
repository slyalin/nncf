"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict

import tensorflow as tf
import tensorflow_datasets as tfds

from openvino.runtime import Core
import torch

from nncf.experimental import ptq
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.ptq.examples.tensorflow.mobilenet_v2.preprocessing import center_crop

FILE = Path(__file__).resolve()
# Relative path to the `mobilenet_v2` directory.
ROOT = Path(os.path.relpath(FILE.parent, Path.cwd()))
# Path to the directory where the original and quantized models will be saved.
MODEL_DIR = ROOT.joinpath('mobilenet_v2_quantization')
# Path to ImageNet validation dataset.
DATASET_DIR = Path("/local/nn_icv_cv_externalN/omz-training-datasets/tensorflow/tfds/imagenet2012")


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    # Step 1: Prepare TF model.
    original_model = tf.keras.applications.MobileNetV2()

    # Step 2: Create dataset.
    val_dataset = create_val_dataset(batch_size=128)

    # Step 3: Apply quantization algorithm.

    # Define the transformation method. This method should
    # take a data item from the data source and transform it
    # into the model expected input.
    def transform_fn(data_item):
        images, _ = data_item
        return images

    # Wrap framework-specific data source into `NNCFDataLoader` object.
    calibration_dataset = ptq.create_dataloader(val_dataset, transform_fn)

    quantized_model = ptq.quantize(original_model, calibration_dataset)

    # Step 4: Save Tensorflow model.
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    model_name = 'mobilenet_v2'
    tf_quantized_model_path = MODEL_DIR.joinpath(model_name)
    quantized_model.save(tf_quantized_model_path)
    print(f"The quantized model is exported to {tf_quantized_model_path}")


    # Step 5: Run Model Optimizer to convert Tensorflow model to OpenVINO IR.
    mo_command = f'mo --saved_model_dir {tf_quantized_model_path} --model_name {model_name} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    # Step 6: Laod IR model.
    ir_model_xml = MODEL_DIR.joinpath(f'{model_name}.xml')
    ir_model_bin = MODEL_DIR.joinpath(f'{model_name}.bin')
    ie = Core()
    ir_quantized_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5')
    ]

    # Step 5: Compare the accuracy of the original and quantized models.
    nncf_logger.info('Checking the accuracy of the original model:')

    original_model.compile(metrics=metrics)
    original_model.evaluate(val_dataset)

    nncf_logger.info('Checking the accuracy of the quantized model:')
    quantized_compiled_model = ie.compile_model(ir_quantized_model, device_name='CPU')
    validate(quantized_compiled_model, metrics, val_dataset)


def create_val_dataset(batch_size: int) -> torch.utils.data.Dataset:
    """
    Creates validation ImageNet dataset.

    :return: The `torch.utils.data.Dataset` object.
    """
    val_dataset = tfds.load('imagenet2012', split='validation', data_dir=DATASET_DIR)

    def preprocess(data_item: Dict[str, tf.Tensor]):
        image = data_item['image']
        image = center_crop(image, 224, 32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        label = data_item['label']
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 1000)
        label = tf.reshape(label, [1000])

        return image, label

    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)

    return val_dataset


def validate(model, metrics, val_dataset, print_freq: int = 10):
    num_items = len(val_dataset)
    for i, (images, labels) in enumerate(val_dataset):
        input_data = images.numpy()

        logit = model(input_data)
        pred = list(logit.values())[0]

        for m in metrics:
            m.update_state(labels, pred)
        if i % print_freq == 0 or i + 1 == num_items:
            print(f'{i + 1}/{num_items}: acc@1: {metrics[0].result().numpy()} acc@5: {metrics[1].result().numpy()}')


if __name__ == '__main__':
    run_example()
