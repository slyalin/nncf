# Quantize the fine-tuned BERT model and check F1 score

We show in this example how to quantize the fine-tuned BERT model using simplified PTQ API. We use the fine-tined BERT model (`bert-base-uncased` model in HuggingFace transformers) for the MRPC task. You can find it [here](https://download.pytorch.org/tutorial/MRPC.zip).

You should do the following steps to work with this example:

1. Download the fine-tuned BERT model to the `${NNCF_ROOT}/nncf/ptq/examples/bert` folder.
   Run the following command to do this:

   ```bash
   NNCF_ROOT="" # absolute path to the NNCF repository
   cd ${NNCF_ROOT}/nncf/ptq/examples/bert
   wget https://download.pytorch.org/tutorial/MRPC.zip
   sudo apt install unzip
   unzip MRPC.zip
   ```

2. Install additional requirements. We assume that `openvino-dev` and `nncf` have been already installed.
   ```bashx
   pip isntall -r requirements.txt
   ```

3. Run the example:

   ```bash
   cd ${NNCF_ROOT}/nncf/ptq/examples/bert
   python bert_quantization.py
   ```
