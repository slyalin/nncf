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
import datasets
import evaluate
import transformers
import numpy as np

from nncf import ptq
from nncf.common.utils.logger import logger as nncf_logger


FILE = Path(__file__).resolve()
# Relative path to the `bert` directory.
ROOT = Path(os.path.relpath(FILE.parent, Path.cwd()))
# Path to the directory where the original and quantized IR will be saved.
MODEL_DIR = ROOT.joinpath('bert_quantization')
# Path to the pre-trained model directory.
PRETRAINED_MODEL_DIR = ROOT.joinpath('MRPC')

TASK_NAME = 'mrpc'
MAX_SEQ_LENGTH = 128


def run_example():
    """
    Runs the BERT quantization example.
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
    INPUT_NAMES = [x.any_name for x in original_model.inputs]
    def transform_fn(data_item):
        inputs = {
            name: np.asarray(data_item[name], dtype=np.int64) for name in INPUT_NAMES
        }
        return inputs

    # Wrap framework-specific data source into `NNCFDataLoader` object.
    calibration_dataset = ptq.create_dataloader(data_source, transform_fn)

    quantized_model = ptq.quantize(original_model, calibration_dataset, model_type='transformer')

    # Step 4: Save quantized model.
    model_name = 'bert_base_mrpc_quantized'
    ir_qmodel_xml = MODEL_DIR.joinpath(f'{model_name}.xml')
    ir_qmodel_bin = MODEL_DIR.joinpath(f'{model_name}.bin')
    openvino.offline_transformations.serialize(quantized_model, str(ir_qmodel_xml), str(ir_qmodel_bin))

    # Step 5: Compare the accuracy of the original and quantized models.
    nncf_logger.info('Checking the accuracy of the original model:')
    original_compiled_model = ie.compile_model(original_model, device_name='CPU')
    validate(original_compiled_model, data_source)

    nncf_logger.info('Checking the accuracy of the quantized model:')
    quantized_compiled_model = ie.compile_model(quantized_model, device_name='CPU')
    validate(quantized_compiled_model, data_source)

    # Step 6: Compare Performance of the original and quantized models.
    # benchmark_app -m bert_quantization/bert_base_mrpc.xml -d CPU -api async
    # benchmark_app -m bert_quantization/bert_base_mrpc_quantized.xml -d CPU -api async


def torch_to_openvino_model() -> Tuple[Path, Path]:
    """
    Converts the fine-tuned BERT model for the MRPC task to the OpenVINO IR format.

    :return: A tuple (ir_model_xml, ir_model_bin) where
        `ir_model_xml` - path to .xml file.
        `ir_model_bin` - path to .bin file.
    """
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    # Step 1: Load pre-trained model.
    model = transformers.BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR)
    model.eval()

    # Step 2: Export PyTorch model to ONNX format.
    model_name = 'bert_base_mrpc'
    onnx_model_path = MODEL_DIR.joinpath(f'{model_name}.onnx')
    dummy_input = (
        torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64),  # input_ids
        torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64),  # attention_mask
        torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64),  # token_type_ids
    )
    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      verbose=False,
                      opset_version=11,
                      input_names=['input_ids', 'attention_mask', 'token_type_ids'],
                      output_names=['output'])

    # Step 3: Run Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = f'mo --framework onnx -m {onnx_model_path} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    # Step 4: Return path to IR model as result.
    ir_model_xml = MODEL_DIR.joinpath(f'{model_name}.xml')
    ir_model_bin = MODEL_DIR.joinpath(f'{model_name}.bin')
    return ir_model_xml, ir_model_bin


def create_val_dataset() -> datasets.Dataset:
    """
    Creates validation MRPC dataset.

    :return: The `datasets.Dataset` object.
    """
    raw_dataset = datasets.load_dataset('glue', TASK_NAME, split='validation')
    tokenizer = transformers.BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)

    def _preprocess_fn(examples):
        texts = (examples['sentence1'], examples['sentence2'])
        result = tokenizer(*texts, padding='max_length', max_length=MAX_SEQ_LENGTH, truncation=True)
        result['labels'] = examples['label']
        return result
    processed_dataset = raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)

    return processed_dataset


def validate(model: openvino.runtime.Model, data_source: datasets.Dataset) -> float:
    """
    Validates the model on the dataset.

    :param model: A model to be validated.
    :param data_source: Dataset for validation.
    :return: F1 score.
    """
    metric = evaluate.load('glue', TASK_NAME)
    INPUT_NAMES = [x.any_name for x in model.inputs]
    output_layer = next(iter(model.outputs))
    for batch in data_source:
        inputs = [
            np.expand_dims(np.asarray(batch[key], dtype=np.int64), 0) for key in INPUT_NAMES
        ]
        outputs = model(inputs)[output_layer]
        predictions = outputs[0].argmax(axis=-1)
        predictions, references = predictions, batch['labels']
        metric.add_batch(predictions=[predictions], references=[references])
    metrics = metric.compute()

    f1_score = metrics['f1']
    nncf_logger.info(f'F1 score: {f1_score}')
    return f1_score


if __name__ == '__main__':
    run_example()
