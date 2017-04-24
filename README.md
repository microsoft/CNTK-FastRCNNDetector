# CNTK-FastRCNNDetector
A python implementation for a CNTK Fast-RCNN evaluation client.

Call a Fast-RCNN python model from your python code, or run as a script directly from the command line.

For more information regarding the CNTK Fast-RCNN implementation, please checkout <a href="https://github.com/Microsoft/CNTK/wiki/Object-Detection-using-Fast-R-CNN">this tutorial</a>.

A detailed notebook containing a walkthrough for evaluating a single image using a Fast-RCNN model, is <a href="https://github.com/nadavbar/cntk-fastrcnn/blob/master/CNTK_FastRCNN_Eval.ipynb">available here</a>. 

In addition, there is also a node.js wrapper for this code that lets you call this code from node.js or Electron: https://github.com/nadavbar/node-cntk-fastrcnn.

## Preliminaries

Since the FRCNN detector uses bits of the CNTK Fast-RCNN implementation it has the same requirements as the CNTK
Fast-RCNN training pipeline. 

Before running the code in this repository, please make sure to install the required python packages as described
in <a href="https://github.com/Microsoft/CNTK/wiki/Object-Detection-using-Fast-R-CNN#setup">the Fast-RCNN CNTK tutorial</a>.  

## Using directly from your python code

In order to use directly from your python code, import frcnn_detector.py and initialize a new 
FRCNNDetector object with a path to your model file and to the CNTK installation.
Then, use the **detect** method to call the model on a given image.

For example, the following code snippet runs detection on a single image and prints the resulting bounding boxes
and the corresponding labels:
```python
import cv2
from os import path
from frcnn_detector import FRCNNDetector

cntk_scripts_path = r'C:/local/cntk/Examples/Image/Detection/FastRCNN'
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
<li><b>load_model()</b> - Loads the model. Note that the detect() method will make sure that the model is loaded in case
the load_model method wasn't called yet.</li>

<li><b>warm_up()</b> - Runs a "dummy" detection through the network. Can be used to make sure that all of the CNTK libraries are
loaded before the actual detection is called.</li>

<li><b>resize_and_pad(img)</b> - Accepts an image in an OpenCV format and resizes (and pads) the image according to the input format that the network accepts.
Returns a tuple of the resized image in an OpenCV readable format, and in the format expected by the network (BGR).</li>

<li><b>get_rois_for_image(img)</b>  - Accepts an image in an OpenCV format and calculates a list of ROIs according to the input format that the network accepts.
As an optimization,tThe grid ROIs are calculated only once and then cached and reused. The method returns a tuple, where the first item is a list of ROIs that correspond 
to the internal network format (in relative image coordinates), and the second item is a list of corresponding ROIs in the format of the original image.</li>
</il>

## Run as a script

The script accepts either a single image or directory of images and outputs either corresponding images 
with highlighted bounding boxes or a JSON file with a textual description of the detection result. (JSON description is available below)  

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
                        Path to the directory in which CNTK is installed, e.g.
                        c:\local\cntk
  --json-output <file path>
                        Path to output JSON file
```

Here is an example of the result object of a directory that contains 2 images (named '1.jpg' and '2.jpg'):
```json
{
	"frames": {
		"1.jpg": {
			"regions": [
				{
					"class": 1,
					"x1": 418,
					"x2": 538,
					"y2": 179,
					"y1": 59
				}
			]
		},
		"2.jpg": {
			"regions": [
				{
					"class": 2,
					"x1": 478,
					"x2": 597,
					"y2": 298,
					"y1": 59
				}
			]
		}
	},
	"classes": {
		"background" : 0,
		"human": 1,
		"cat": 2,
		"dog" : 3
	}
}
```

### Adding descriptive classes names
Since CNTK does not embed the names of the classes in the model, on default, the module returns non descriptive names for the classes, e.g. "class_1", "class_2".

If you want the module to return more descriptive names, you can place a JSON file named "model.json" in the same directory of the Fast-RCNN model file.
You can then place the descriptions of the classes in the JSON file under the "classes" key.

For example, the following JSON will describe the classes for the above example:

```json
{
    "classes" : {
        "background" : 0,
        "human" : 1,
		"cat" : 2,
		"dog" : 3
    }
}
```
