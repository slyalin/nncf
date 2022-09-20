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
from typing import Tuple

import openvino
from openvino.runtime import Core
import torch

from nncf import ptq
from nncf.common.utils.logger import logger as nncf_logger

from yolov5.utils.general import check_dataset
from yolov5.utils.general import download
from yolov5.utils.dataloaders import create_dataloader
from yolov5.val import run as validate


FILE = Path(__file__).resolve()
# Relative path to the `yolo_v5` directory.
ROOT = Path(os.path.relpath(FILE.parent, Path.cwd()))
# Path to the directory where the original and quantized IR will be saved.
MODEL_DIR = ROOT.joinpath('yolov5m_quantization')
# Path to the dataset config from the `ultralytics/yolov5` repository.
DATASET_CONFIG = ROOT.joinpath('yolov5', 'data', 'coco.yaml')


def run_example():
    """
    Runs the YOLOv5 quantization example.
    """
    # Step 1: Prepare OpenVINO model.
    ir_model_xml, ir_model_bin = torch_to_openvino_model()
    ie = Core()
    original_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    # Step 2: Create dataset.
    data_source = create_val_dataset()

    # Step 3: Apply quantization algorithm.

    # Define the transformation method. This method should
    # take a data item from the data source and transform it
    # into the model expected input.
    def transform_fn(data_item):
        images, *_ = data_item
        images = images.float()
        images = images / 255
        images = images.cpu().detach().numpy()
        return images

    # Wrap framework-specific data source into `NNCFDataLoader` object.
    calibration_dataset = ptq.create_dataloader(data_source, transform_fn)

    quantized_model = ptq.quantize(original_model, calibration_dataset, preset='mixed')

    # Step 4: Save quantized model.
    model_name = 'yolov5m_quantized'
    ir_qmodel_xml = MODEL_DIR.joinpath(f'{model_name}.xml')
    ir_qmodel_bin = MODEL_DIR.joinpath(f'{model_name}.bin')
    openvino.offline_transformations.serialize(quantized_model, str(ir_qmodel_xml), str(ir_qmodel_bin))

    # Step 5: Compare the accuracy of the original and quantized models.
    nncf_logger.info('Checking the accuracy of the original model:')
    metrics = validate(data=DATASET_CONFIG,
                       weights=ir_model_xml,  # Already supports.
                       batch_size=1,
                       workers=1,
                       plots=False,
                       device='cpu',
                       iou_thres=0.65)
    nncf_logger.info(f'mAP@.5 = {metrics[0][2]}')
    nncf_logger.info(f'mAP@.5:.95 = {metrics[0][3]}')

    nncf_logger.info('Checking the accuracy of the quantized model:')
    metrics = validate(data=DATASET_CONFIG,
                       weights=ir_qmodel_xml,  # Already supports.
                       batch_size=1,
                       workers=1,
                       plots=False,
                       device='cpu',
                       iou_thres=0.65)
    nncf_logger.info(f'mAP@.5 = {metrics[0][2]}')
    nncf_logger.info(f'mAP@.5:.95 = {metrics[0][3]}')

    # Step 6: Compare Performance of the original and quantized models
    # benchmark_app -m yolov5m_quantization/yolov5m.xml -d CPU -api async
    # benchmark_app -m yolov5m_quantization/yolov5m_quantized.xml -d CPU -api async


def torch_to_openvino_model() -> Tuple[Path, Path]:
    """
    Converts PyTorch YOLOv5 model to the OpenVINO IR format.

    :return: A tuple (ir_model_xml, ir_model_bin) where
        `ir_model_xml` - path to .xml file.
        `ir_model_bin` - path to .bin file.
    """
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    # Step 1: Export PyTorch model to ONNX format. We use the export script
    # from the `ultralytics/yolov5` repository.
    model_name = 'yolov5m'
    yolov5_repo_path = ROOT.joinpath('yolov5')
    export_command = (
        f'python {yolov5_repo_path}/export.py '
        f'--weights {yolov5_repo_path}/{model_name}/{model_name}.pt '
        '--imgsz 640 '
        '--batch-size 1 '
        '--include onnx'
    )
    subprocess.call(export_command, shell=True)
    onnx_model_path = yolov5_repo_path.joinpath(model_name, f'{model_name}.onnx')

    # Step 2: Run Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = f'mo --framework onnx -m {onnx_model_path} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    # Step 3: Return path to IR model as result.
    ir_model_xml = MODEL_DIR.joinpath(f'{model_name}.xml')
    ir_model_bin = MODEL_DIR.joinpath(f'{model_name}.bin')
    return ir_model_xml, ir_model_bin


def create_val_dataset() -> torch.utils.data.Dataset:
    """
    Creates COCO 2017 validation dataset. The method downloads COCO 2017
    dataset if it does not exist.

    :return: The `torch.utils.data.Dataset` object.
    """
    if not ROOT.joinpath('datasets', 'coco').exists():
        urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip']
        download(urls, dir=ROOT.joinpath('datasets'))

        urls = ['http://images.cocodataset.org/zips/val2017.zip']
        download(urls, dir=ROOT.joinpath('datasets', 'coco', 'images'))

    data = check_dataset(DATASET_CONFIG)
    val_dataset = create_dataloader(data['val'],
                                    imgsz=640,
                                    batch_size=1,
                                    stride=32,
                                    pad=0.5,
                                    workers=1)[1]

    return val_dataset


if __name__ == '__main__':
    run_example()
