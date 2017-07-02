import cv2
import json
import sys
import numpy as np
from os import path

# CNTK imports
from cntk import load_model
from cntk import placeholder
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.ops import combine
from cntk.ops.sequence import input_variable
from cntk.ops.functions import CloneMethod

# constants used for ROI generation:
# ROI generation
roi_minDimRel = 0.04
roi_maxDimRel = 0.4
roi_minNrPixelsRel = 2 * roi_minDimRel * roi_minDimRel
roi_maxNrPixelsRel = 0.33 * roi_maxDimRel * roi_maxDimRel
roi_maxAspectRatio = 7.0  # maximum aspect Ratio of a ROI vertically and horizontally
roi_maxImgDim = 200  # image size used for ROI generation
ss_scale = 100  # selective search ROIS: parameter controlling cluster size for segmentation
ss_sigma = 1.2  # selective search ROIs: width of gaussian kernal for segmentation
ss_minSize = 20  # selective search ROIs: minimum component size for segmentation
grid_nrScales = 7  # uniform grid ROIs: number of iterations from largest possible ROI to smaller ROIs
grid_aspectRatios = [0.5, 4.0, 6.0]  # uniform grid ROIs: aspect ratio of ROIs
roi_minDim = roi_minDimRel * roi_maxImgDim
roi_maxDim = roi_maxDimRel * roi_maxImgDim
roi_minNrPixels = roi_minNrPixelsRel * roi_maxImgDim * roi_maxImgDim
roi_maxNrPixels = roi_maxNrPixelsRel * roi_maxImgDim * roi_maxImgDim
nms_threshold = 0.1


def get_classes_description(model_file_path, classes_count):
    model_dir = path.dirname(model_file_path)
    classes_names = {}
    model_desc_file_path = path.join(model_dir, 'model.json')
    if not path.exists(model_desc_file_path):
        # use default parameter names:
        for i in range(classes_count):
            classes_names["class_%d"%i] = i
        return classes_names
    
    with open(model_desc_file_path) as handle:
        file_content = handle.read()
        model_desc = json.loads(file_content)
    return model_desc["classes"]


class FRCNNDetector:

    def __init__(self, model_path,
                 pad_value = 114, cntk_scripts_path=r"c:\local\cntk\Examples\Image\Detection\FastRCNN",
                 use_selective_search_rois = True,
                 use_grid_rois = True):
        self.__model_path = model_path
        self.__cntk_scripts_path = cntk_scripts_path
        self.__pad_value = pad_value
        self.__pad_value_rgb = [pad_value, pad_value, pad_value]
        self.__use_selective_search_rois = use_selective_search_rois
        self.__use_grid_rois = use_grid_rois
        self.__model = None
        self.__isPythonModel = None
        self.__model_warm = False
        self.__grid_rois_cache = {}

        self.labels_count = 0

        # a cache to use ROIs after filter in case we only use the grid method
        self.__rois_only_grid_cache = {}

        sys.path.append(self.__cntk_scripts_path)
        global imArrayWidthHeight, getSelectiveSearchRois, imresizeMaxDim
        from cntk_helpers import imArrayWidthHeight, getSelectiveSearchRois, imresizeMaxDim
        global getGridRois, filterRois, roiTransformPadScaleParams, roiTransformPadScale
        from cntk_helpers import getGridRois, filterRois, roiTransformPadScaleParams, roiTransformPadScale
        global softmax2D, applyNonMaximaSuppression
        from cntk_helpers import softmax2D, applyNonMaximaSuppression

    def ensure_model_is_loaded(self):
        if not self.__model:
            self.load_model()

    def warm_up(self):
        self.ensure_model_is_loaded()

        if self.__model_warm:
            return

        # a dummy variable for labels the will be given as an input to the network but will be ignored
        dummy_labels = np.zeros((self.__nr_rois, self.labels_count))
        dummy_rois = np.zeros((self.__nr_rois, 4))
        dummy_image = np.ones((3, self.__resize_width, self.__resize_height)) * 255.0

        # prepare the arguments
        if (self.__isPythonModel):#python model
            arguments = {
                self.__model.arguments[0]: [dummy_image],
                self.__model.arguments[1]: [dummy_rois]
            }
        else: #brainscript
            arguments = {
                self.__model.arguments[self.__args_indices["features"]]: [dummy_image],
                self.__model.arguments[self.__args_indices["rois"]]: [dummy_rois]
            }

        self.__model.eval(arguments)

        self.__model_warm = True


    def load_model(self):
        if self.__model:
            raise Exception("Model already loaded")
        
        trained_frcnn_model = load_model(self.__model_path)
        self.__isPythonModel = (True if (len(trained_frcnn_model.arguments) < 3) else False)

        if (self.__isPythonModel):#python model
            self.__nr_rois = trained_frcnn_model.arguments[1].shape[0]
            self.__resize_width = trained_frcnn_model.arguments[0].shape[1]
            self.__resize_height = trained_frcnn_model.arguments[0].shape[2]
            self.labels_count = 4 # this should be the number of labels in the model 
            self.__model = trained_frcnn_model

        else: #brainscript
            # cache indices of the model arguments
            args_indices = {}
            for i,arg in enumerate(trained_frcnn_model.arguments):
               args_indices[arg.name] = i

            self.__nr_rois = trained_frcnn_model.arguments[args_indices["rois"]].shape[0]
            self.__resize_width = trained_frcnn_model.arguments[args_indices["features"]].shape[1]
            self.__resize_height = trained_frcnn_model.arguments[args_indices["features"]].shape[2]
            self.labels_count = trained_frcnn_model.arguments[args_indices["roiLabels"]].shape[1]
            
            # next, we adjust the clone the model and create input nodes just for the features (image) and ROIs
            # This will make sure that only the calculations that are needed for evaluating images are performed
            # during test time
            #  
            # find the original features and rois input nodes
            features_node = find_by_name(trained_frcnn_model, "features")
            rois_node = find_by_name(trained_frcnn_model, "rois")

            #  find the output "z" node
            z_node = find_by_name(trained_frcnn_model, 'z')

            # define new input nodes for the features (image) and rois
            image_input = input_variable(features_node.shape, name='features')
            roi_input = input_variable(rois_node.shape, name='rois')

            # Clone the desired layers with fixed weights and place holder for the new input nodes
            cloned_nodes = combine([z_node.owner]).clone(
               CloneMethod.freeze,
               {features_node: placeholder(name='features'), rois_node: placeholder(name='rois')})

            # apply the cloned nodes to the input nodes to obtain the model for evaluation
            self.__model = cloned_nodes(image_input, roi_input)

        # cache the indices of the input nodes
        self.__args_indices = {}

        for i,arg in enumerate(self.__model.arguments):
            self.__args_indices[arg.name] = i


    def resize_and_pad(self, img):
        self.ensure_model_is_loaded()

        # port of the c++ code from CNTK: https://github.com/Microsoft/CNTK/blob/f686879b654285d06d75c69ee266e9d4b7b87bc4/Source/Readers/ImageReader/ImageTransformers.cpp#L316
        img_width = len(img[0])
        img_height = len(img)

        scale_w = img_width > img_height

        target_w = self.__resize_width
        target_h = self.__resize_height

        if scale_w:
            target_h = int(np.round(img_height * float(self.__resize_width) / float(img_width)))
        else:
            target_w = int(np.round(img_width * float(self.__resize_height) / float(img_height)))

        resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)

        top = int(max(0, np.round((self.__resize_height - target_h) / 2)))
        left = int(max(0, np.round((self.__resize_width - target_w) / 2)))

        bottom = self.__resize_height - top - target_h
        right = self.__resize_width - left - target_w

        resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                              cv2.BORDER_CONSTANT, value=self.__pad_value_rgb)

        # tranpose(2,0,1) converts the image to the HWC format which CNTK accepts
        model_arg_rep = np.ascontiguousarray(np.array(resized_with_pad, dtype=np.float32).transpose(2, 0, 1))

        return resized_with_pad, model_arg_rep

    def get_rois_for_image(self, img):
        self.ensure_model_is_loaded()

        # get rois
        if self.__use_selective_search_rois:
            rects, scaled_img, scale = getSelectiveSearchRois(img, ss_scale, ss_sigma, ss_minSize,
                                                              roi_maxImgDim)  # interpolation=cv2.INTER_AREA
        else:
            rects = []
            scaled_img, scale = imresizeMaxDim(img, roi_maxImgDim, boUpscale=True, interpolation=cv2.INTER_AREA)

        imgWidth, imgHeight = imArrayWidthHeight(scaled_img)

        if not self.__use_selective_search_rois:
            if (imgWidth, imgHeight) in self.__rois_only_grid_cache:
                return self.__rois_only_grid_cache[(imgWidth, imgHeight)]

        # add grid rois
        if self.__use_grid_rois:
            if (imgWidth, imgHeight) in self.__grid_rois_cache:
                rectsGrid = self.__grid_rois_cache[(imgWidth, imgHeight)]
            else:
                rectsGrid = getGridRois(imgWidth, imgHeight, grid_nrScales, grid_aspectRatios)
                self.__grid_rois_cache[(imgWidth, imgHeight)] = rectsGrid

            rects += rectsGrid

        # run filter
        rois = filterRois(rects, imgWidth, imgHeight, roi_minNrPixels, roi_maxNrPixels, roi_minDim, roi_maxDim,
                              roi_maxAspectRatio)
        if len(rois) == 0:  # make sure at least one roi returned per image
            rois = [[5, 5, imgWidth - 5, imgHeight - 5]]

        # scale up to original size and save to disk
        # note: each rectangle is in original image format with [x,y,x2,y2]
        original_rois = np.int32(np.array(rois) / scale)

        img_width = len(img[0])
        img_height = len(img)

        # all rois need to be scaled + padded to cntk input image size
        targetw, targeth, w_offset, h_offset, scale = roiTransformPadScaleParams(img_width, img_height,
                                                                                 self.__resize_width,
                                                                                 self.__resize_height)

        rois = []
        for original_roi in original_rois:
            x, y, x2, y2 = roiTransformPadScale(original_roi, w_offset, h_offset, scale)

            xrel = float(x) / (1.0 * targetw)
            yrel = float(y) / (1.0 * targeth)
            wrel = float(x2 - x) / (1.0 * targetw)
            hrel = float(y2 - y) / (1.0 * targeth)

            rois.append([xrel, yrel, wrel, hrel])

        # pad rois if needed:
        if len(rois) < self.__nr_rois:
            rois += [[0, 0, 0, 0]] * (self.__nr_rois - len(rois))
        elif len(rois) > self.__nr_rois:
            rois = rois[:self.__nr_rois]

        if not self.__use_selective_search_rois:
            self.__rois_only_grid_cache[(imgWidth, imgHeight)] = (np.array(rois), original_rois)
        return np.array(rois), original_rois

    def detect(self, img):

        self.ensure_model_is_loaded()
        self.warm_up()

        resized_img, img_model_arg = self.resize_and_pad(img)

        test_rois, original_rois = self.get_rois_for_image(img)

        roi_padding_index = len(original_rois)

        # a dummy variable for labels the will be given as an input to the network but will be ignored
        dummy_labels = np.zeros((self.__nr_rois, self.labels_count))

        # prepare the arguments
        if (self.__isPythonModel):#python model
            arguments = {
                self.__model.arguments[0]: [img_model_arg],
                self.__model.arguments[1]: [test_rois]
            }
        else: #brainscript
            arguments = {
                self.__model.arguments[self.__args_indices["features"]]: [img_model_arg],
                self.__model.arguments[self.__args_indices["rois"]]: [test_rois]
            }

        # run it through the model
        output = self.__model.eval(arguments)
        self.__model_warm  = True
        
        # take just the relevant part and cast to float64 to prevent overflow when doing softmax
        if (self.__isPythonModel):#python model
            rois_values = output[0][:roi_padding_index].astype(np.float64)
            print("===", rois_values.shape)
        else: #brainscript
            rois_values = output[0][0][:roi_padding_index].astype(np.float64)

        # get the prediction for each roi by taking the index with the maximal value in each row
        rois_labels_predictions = np.argmax(rois_values, axis=1)

        # calculate the probabilities using softmax
        rois_probs = softmax2D(rois_values)

        non_padded_rois = test_rois[:roi_padding_index]
        max_probs = np.amax(rois_probs, axis=1).tolist()

        rois_prediction_indices = applyNonMaximaSuppression(nms_threshold, rois_labels_predictions, max_probs,
                                                            non_padded_rois)

        original_rois_predictions = original_rois[rois_prediction_indices]

        rois_predictions_labels = rois_labels_predictions[rois_prediction_indices]
        # filter out backgrond label
        non_background_indices = rois_predictions_labels > 0
        rois_predictions_labels = rois_predictions_labels[non_background_indices]
        rois_predictions = original_rois_predictions[non_background_indices]
        return rois_predictions, rois_predictions_labels


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='FRCNN Detector')
    
    parser.add_argument('--input', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing image in jpg format', required=True)
    
    parser.add_argument('--output', type=str, metavar='<directory path>',
                        help='Path to output directory', required=False)
    
    parser.add_argument('--model', type=str, metavar='<file path>',
                        help='Path to model file',
                        required=True)

    parser.add_argument('--cntk-path', type=str, metavar='<dir path>',
                        help='Path to the directory in which CNTK is installed, e.g. c:\\local\\cntk',
                        required=False)

    parser.add_argument('--json-output', type=str, metavar='<file path>',
                        help='Path to output JSON file', required=False)

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    json_output_path = args.json_output
    model_file_path = args.model

    if args.cntk_path:
        cntk_path = args.cntk_path
    else:
        cntk_path = "C:\\local\\cntk"
    cntk_scripts_path = path.join(cntk_path, r"Examples/Image/Detection/FastRCNN")

    if (output_path is None and json_output_path is None):
        parser.error("No directory output path or json output path specified")

    if (output_path is not None) and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if os.path.isdir(input_path):
        import glob
        file_paths = glob.glob(os.path.join(input_path, '*.jpg'))
    else:
        file_paths = [input_path]

    detector = FRCNNDetector(model_file_path, use_selective_search_rois=False, 
                            cntk_scripts_path=cntk_scripts_path)
    detector.load_model()

    if json_output_path is not None:
        model_classes = get_classes_description(model_file_path, detector.labels_count)
        json_output_obj = {"classes": model_classes,
                           "frames" : {}}

    colors = [(0,0,0), (255,0,0), (0,0,255)]
    players_label = -1
    print("Number of images to process: %d"%len(file_paths))

    for file_path, counter in zip(file_paths, range(len(file_paths))):
        print("Read file in path:", file_path)
        img = cv2.imread(file_path)
        rects, labels = detector.detect(img)

        print("Processed image %d"%(counter+1))

        if output_path is not None:
            img_cpy = img.copy()

            print("Running FRCNN detection on", file_path)
            print("%d regions were detected"%len(rects))

            for rect, label in zip(rects, labels):
                x1, y1, x2, y2 = rect

                cv2.rectangle(img_cpy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output_file_path = os.path.join(output_path, os.path.basename(file_path))
            cv2.imwrite(output_file_path, img_cpy)
        elif json_output_path is not None:
            image_base_name = path.basename(file_path)
            regions_list = []
            json_output_obj["frames"][image_base_name] = {"regions": regions_list}
            for rect, label in zip(rects, labels):
                regions_list.append({
                    "x1" : int(rect[0]),
                    "y1" : int(rect[1]),
                    "x2" : int(rect[2]),
                    "y2" : int(rect[3]),
                    "class" : int(label)
                })

    if json_output_path is not None:
        with open(json_output_path, "wt") as handle:
            json_dump = json.dumps(json_output_obj, indent=2)
            handle.write(json_dump)
