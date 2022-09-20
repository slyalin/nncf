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
from pathlib import Path
import time

import onnx
import onnxruntime as rt
import torch
import torchvision
import numpy as np

from nncf.experimental import ptq
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.ptq.examples.openvino.mobilenet_v2 import usercode


FILE = Path(__file__).resolve()
# Relative path to the `mobilenet_v2` directory.
ROOT = Path(os.path.relpath(FILE.parent, Path.cwd()))
# Path to the directory where the original and quantized models will be saved.
MODEL_DIR = ROOT.joinpath('mobilenet_v2_quantization')
# Path to ImageNet validation dataset.
DATASET_DIR = Path("/home/susloval/dataset/imagenet")


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    # Step 1: Prepare ONNX model.
    onnx_model_path = torch_to_onnx_model()
    original_model = onnx.load(onnx_model_path)

    # Step 2: Create dataset.
    data_source = create_val_dataset()

    # Step 3: Apply quantization algorithm.

    # Define the transformation method. This method should
    # take a data item from the data source and transform it
    # into the model expected input.
    input_name = original_model.graph.input[0].name
    def transform_fn(data_item):
        images, _ = data_item
        return {input_name: images.numpy()}

    # Wrap framework-specific data source into `NNCFDataLoader` object.
    calibration_dataset = ptq.create_dataloader(data_source, transform_fn)

    quantized_model = ptq.quantize(original_model, calibration_dataset)

    # Step 4: Save quantized model.
    quantized_model_path = MODEL_DIR.joinpath('quantized_mobilenet_v2.onnx')
    onnx.save(quantized_model, quantized_model_path)
    print(f"The quantized model is saved on {quantized_model_path}")

    # Step 5: Compare the accuracy of the original and quantized models.
    nncf_logger.info('Checking the accuracy of the original model:')
    validate(original_model,
             data_source,
             providers=['OpenVINOExecutionProvider'],
             provider_options=[{'device_type' : 'CPU_FP32'}])

    nncf_logger.info('Checking the accuracy of the quantized model:')
    validate(quantized_model,
             data_source,
             providers=['OpenVINOExecutionProvider'],
             provider_options=[{'device_type' : 'CPU_FP32'}])


def torch_to_onnx_model() -> Path:
    """
    Converts PyTorch MobileNetV2 model to the ONNX format.

    :return: path to .onnx file.
    """
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    # Step 1: Initialize model from the PyTorch Hub.
    # For more details, please see the [link](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    model_name = 'mobilenet_v2'
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    model.eval()

    # Step 2: Export PyTorch model to ONNX format.
    onnx_model_path = MODEL_DIR.joinpath(f'{model_name}.onnx')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    return onnx_model_path


def create_val_dataset() -> torch.utils.data.Dataset:
    """
    Creates validation ImageNet dataset.

    :return: The `torch.utils.data.Dataset` object.
    """
    val_dir = DATASET_DIR.joinpath('val')
    # Transformations were taken from [here](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, preprocess)

    return val_dataset


def validate(model,
             val_loader,
             providers=['OpenVINOExecutionProvider'],
             provider_options=[{'device_type' : 'CPU_FP32'}],
             print_freq: int = 10000):

    def run_validate(sess, input_name, output_names, loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                target = torch.from_numpy(np.expand_dims(np.array([target]), 0))
                input_data = np.expand_dims(images.numpy(), 0).astype(np.float32)
                output = torch.from_numpy(
                    sess.run(output_names, {input_name: input_data})[0]
                )

                # measure accuracy
                acc1, acc5 = usercode.accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i + 1)

    batch_time = usercode.AverageMeter('Time', ':6.3f', usercode.Summary.NONE)
    top1 = usercode.AverageMeter('Acc@1', ':6.2f', usercode.Summary.AVERAGE)
    top5 = usercode.AverageMeter('Acc@5', ':6.2f', usercode.Summary.AVERAGE)

    progress = usercode.ProgressMeter(
        len(val_loader), [batch_time, top1, top5], prefix='Test: '
    )

    so = rt.SessionOptions()
    so.log_severity_level = 3
    sess = rt.InferenceSession(model.SerializeToString(), so, providers, provider_options)
    input_name = sess.get_inputs()[0].name
    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))

    run_validate(sess, input_name, output_names, val_loader)
    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    run_example()
