import torch
import numpy as np
import open3d as o3d
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin
from segment_anything import SamPredictor, sam_model_registry
import os
from PIL import Image

import matplotlib
# somehow adding this for loop resolves this issue: https://github.com/isl-org/Open3D/issues/1715
matplotlib.use('agg')

class VLM():
    """
    Copied from GELLO
    """
    def __init__(self):
        device = torch.device("cpu")
        self.device = device

        self.image_name_counter = 0

        self.model_owl_vit = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model_owl_vit = self.model_owl_vit.to(device)
        self.model_owl_vit.eval()

        if os.path.abspath(os.getcwd()).split('/')[-1] == 'tools':
            # user is using dataset_generator script
            checkpoint = '../../peract/data/segment_anything_vit_h.pth'
            self.debug_output_file_path = '../../peract/outputs/'
        else:
            checkpoint = '../../../../data/segment_anything_vit_h.pth'
            self.debug_output_file_path = './'
        self.model_segment_anything = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.model_segment_anything.to(device=device)
        self.model_segment_anything_predictor = SamPredictor(self.model_segment_anything)

    def _plot_predictions(self, image, score, box, text_queries):
        input_image = np.asarray(image).astype(np.float32) / 255.0

        fig, ax = plt.subplots()
        ax.imshow(input_image)
        ax.set_axis_off()

        cx, cy, w, h = box
        cy = round(cy * image.shape[0])
        h = round(h * image.shape[0])
        cx = round(cx * image.shape[1])
        w = round(w * image.shape[1])
        print(f'Adjusted cx {cx}, cy {cy}, w {w}, h {h}')
        ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f"{text_queries[0]}: {score:1.2f}",
            ha="left",
            va="top",
            color="red",
            bbox={
                "facecolor": "white",
                "edgecolor": "red",
                "boxstyle": "square,pad=.3"
            })
        filename = f'{self.debug_output_file_path}{self.image_name_counter}_debug_owl_vit_predictions.png'
        fig.savefig(filename)
        plt.close()
        print(f'{filename} saved!')

    def _select_best_bbox(self, text_query, scores, boxes):
        best_score_index = np.argmax(scores)
        score = scores[best_score_index]
        box = boxes[best_score_index]
        return score, box

    def get_bounding_box_using_owl_vit(self, text_query, front_rgb, debug):
        """
        Based on OWL-ViT implementation from Colab: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb
        """
        text_queries = [text_query]

        # Process image and text inputs
        inputs = self.model_processor(text=text_queries, images=front_rgb, return_tensors="pt").to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model_owl_vit(**inputs)

        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

        score, box = self._select_best_bbox(text_query, scores, boxes)
        print(f'OWL-ViT prediction score {score} and bounding box {box}')

        if debug:
            self._plot_predictions(front_rgb, score, box, text_queries)

        return box

    def _plot_segment_anything_result(self, front_rgb_np, mask):
        # visualize the predicted masks
        plt.figure(figsize=(10, 10))
        plt.imshow(front_rgb_np)
        plt.imshow(mask, cmap='gray', alpha=0.7)
        y_indices, x_indices = np.nonzero(mask)
        cX = np.mean(x_indices).astype(int)
        cY = np.mean(y_indices).astype(int)
        plt.plot(cX, cY, marker='v', color="red")
        filename = f'{self.debug_output_file_path}{self.image_name_counter}_debug_mask.png'
        plt.savefig(filename)
        plt.close()
        print(f'{filename} saved!')

    def get_segmentation_mask_using_segment_anything(self, bbox, front_rgb, debug):
        """
        Based on Segment Anything tuorial from https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
        """
        # process bounding box into xyxy format (min_x, min_y, max_x, max_y)
        front_rgb_np = np.array(front_rgb)

        # rescale bounding box to rgb's height and width
        cx, cy, w, h = bbox
        cy = round(cy * front_rgb_np.shape[0])
        h = round(h * front_rgb_np.shape[0])
        cx = round(cx * front_rgb_np.shape[1])
        w = round(w * front_rgb_np.shape[1])
        xywh = np.array([cx, cy, w, h], dtype=int)

        xyxy = np.array([round(xywh[0] - (xywh[2]/2)), round(xywh[1] - (xywh[3]/2)), round(xywh[0] + (xywh[2]/2)), round(xywh[1] + (xywh[3]/2))])
        self.model_segment_anything_predictor.set_image(front_rgb_np)

        masks, scores, logits = self.model_segment_anything_predictor.predict(box=xyxy, multimask_output=True)

        best_score_index = np.argmax(scores)
        best_score_mask = masks[best_score_index]

        if debug:
            self._plot_segment_anything_result(front_rgb_np, best_score_mask)

        best_score_mask = best_score_mask.astype(np.bool_)

        return best_score_mask

    def get_target_object_world_coords(self, front_rgb, points, task_name, debug=True):
        # NOTE: modify this when new task is added
        if task_name in ['OpenDrawer', 'open_drawer', 'PutItemInDrawer', 'put_item_in_drawer']:
            text_query = 'top drawer handle'
        elif task_name in ['OpenJar', 'open_jar']:
            text_query = 'jar'
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!! NotImplementedError in get_target_object_world_coords !!!!!!!!!!!!!!!!!!!!!!!')
            raise NotImplementedError

        bbox = self.get_bounding_box_using_owl_vit(text_query, front_rgb, debug)
        mask = self.get_segmentation_mask_using_segment_anything(bbox, front_rgb, debug)
        self.image_name_counter += 1

        obj_points = points[mask]
        if len(obj_points) == 0:
            raise ValueError(f"Object {text_query} not found in the scene")

        # method 1: voxel downsample using o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(obj_points)
        # pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        # obj_points = np.asarray(pcd_downsampled.points)
        # target_object_world_coords = np.mean(obj_points, axis=0)

        # method 2: get the centroid from the mask
        y_indices, x_indices = np.nonzero(mask)
        cX = np.mean(x_indices).astype(int)
        cY = np.mean(y_indices).astype(int)
        target_object_world_coords = points[cY, cX]

        return target_object_world_coords

    def reset_image_name_counter(self):
        self.image_name_counter = 0