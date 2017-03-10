# CNTK-FastRCNNDetector
A python implementation for a CNTK Fast-RCNN evaluation client

The script supports the following cmd line options:

```
usage: frcnn_detector.py [-h] --input <path> [--output <directory path>]
                         --model <file path> [--cntk-path <dir path>]
                         [--json-output <file path>]

FRCNN Detector

optional arguments:
  -h, --help            show this help message and exit
  --input <path>        Path to image file or to a directory containing image
                        in jpg format
  --output <directory path>
                        Path to output directory
  --model <file path>   Path to model file
  --cntk-path <dir path>
                        Path to the diretory in which CNTK is installed, e.g.
                        c:\local\cntk
  --json-output <file path>
                        Path to output JSON file
```