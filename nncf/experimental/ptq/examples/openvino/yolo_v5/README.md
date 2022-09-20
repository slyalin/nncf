# Quantize the Ultralytics YOLOv5 model and check accuracy

You should do the following steps to work with this example:

1. Clone the `yolov5` repository into `${NNCF_ROOT}/nncf/ptq/examples/yolo_v5` folder and install requirements.
   Run the following command to do this:

   ```bash
   NNCF_ROOT="" # absolute path to the NNCF repository
   cd ${NNCF_ROOT}/nncf/ptq/examples/yolo_v5
   git clone https://github.com/ultralytics/yolov5.git -b v6.2
   pip install -r yolov5/requirements.txt
   sudo apt install unzip
   ```

2. Set the `PYTHONPATH` environment variable value to be the path to the `yolov5` directory.
   Run the following command to do this:

   ```bash
   export PYTHONPATH=${PYTHONPATH}:${NNCF_ROOT}/nncf/ptq/examples/yolo_v5/yolov5
   ```

3. Run the example:

   ```bash
   cd ${NNCF_ROOT}/nncf/ptq/examples/yolo_v5
   python yolo_v5_quantization.py
   ```

   You do not need to prepare the COCO 2017 validation dataset. It will be downloaded automatically:

   ```
   yolo_v5
   ├── yolov5
   └── datasets
       └── coco  <- Here
   ```
