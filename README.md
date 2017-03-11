# CNTK-FastRCNNDetector
A python implementation for a CNTK Fast-RCNN evaluation client

Call a Fast-RCNN python model from your python code, or run as a script directly from the command line.

In script mode, the script supports the following cmd line options:

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

In order to use directly from your python code, import frcnn_detector.py and initialize a new 
FRCNNDetector object with a path to your model file and to the CNTK installation.
Then, use the **detect** method to call the model on a given image.

For example, the following code snippet runs detection on a single image and prints the resulting bounding boxes
and the corresponding labels:
```python
import cv2
from os import path
from frcnn_detector import FRCNNDetector

cntk_scripts_path = r'C:\local\cntk\Examples\Image\Detection\FastRCNN'
model_file_path = path.join(cntk_scripts_path, r'proc/grocery_2000/cntkFiles/Output/Fast-RCNN.model')

# initialize the detector and load the model
detector = FRCNNDetector(model_file_path, cntk_scripts_path=cntk_scripts_path)

img = cv2.imread(path.join(cntk_scripts_path,'r../../DataSets/Grocery/testImages/WIN_20160803_11_28_42_Pro.jpg')
rects, labels = detector.detect(img)

# print detections
for rect, label in zip(rects, labels):
    print("Bounding box: %s, label %s"%(rect, label))
```

API Documentation:

The FRCNNDetector constructor accepts the following input parameters:
```
model_path (string) - Path to the Fast-RCNN model file
pad_value (integer) - The value used to pad the resized image (default value is 114)
cntk_scripts_path (string) - Path to the CNTK Fast-RCNN scripts folder. Default value:  r"c:\local\cntk\Examples\Image\Detection\FastRCNN"
use_selective_search_rois (boolean)  - Indicates whether the selective search method should be used when preparing the input ROIs. Default value : True,
use_grid_rois (boolean) - Indicates whether the grid method should be used when preparing the input ROIs. Default value : True
```

The FRCNN detector exposes the set of following method for object detection: 

**detect(img)** - Accepts an image in the OpenCV format and returns a tuple of bounding boxes and labels according to the
FRCNN-Model detection.

Note that all you need is to call the **detect** method
in order to run detection using the model.

The following set of methods are helper methods that you can use in case you need to do anything extra:

<ul>
<li>**load_model()** - Loads the model. Note that the detect() method will make sure that the model is loaded in case
the load_model method wasn't called yet.</li>

<li>**warm_up()** - Runs a "dummy" detection through the network. Can be used to make sure that all of the CNTK libraries are
loaded before the actual detection is called.</li>

<li>**resize_and_pad(img)** - Accepts an image in an OpenCV format and resizes (and padds) the image according to the input format that the network accepts.
Returns a tuple of the resized image in an OpenCV resable format, and in the format expected by the network (BGR).</li>

<li>**get_rois_for_image(img)**  - Accepts an image in an OpenCV format and calculates a list of ROIs according to the input format that the network accepts.
As an optimization,tThe grid ROIs are calculated only once and then cached and reused. The method returns a tuple, where the first item is a list of ROIs that correspond 
to the internal network format (in relative image coordinates), and the second item is a list of correpsonding ROIs in the format of the original image.</li>
</il>