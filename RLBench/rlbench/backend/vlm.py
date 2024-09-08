import torch
import numpy as np
import open3d as o3d
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin
from segment_anything import SamPredictor, sam_model_registry
import os
from PIL import Image

class VLM():

    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
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
        mixin = ImageFeatureExtractionMixin()

        # Load example image
        image_size = self.model_owl_vit.config.vision_config.image_size
        image = mixin.resize(image, image_size)
        input_image = np.asarray(image).astype(np.float32) / 255.0

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

        cx, cy, w, h = box
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
        image_wh = front_rgb_np.shape[0]
        xywh = np.array(np.round(bbox * image_wh), dtype=int)
        xyxy = np.array([round(xywh[0] - (xywh[2]/2)), round(xywh[1] - (xywh[3]/2)), round(xywh[0] + (xywh[2]/2)), round(xywh[1] + (xywh[3]/2))])
        self.model_segment_anything_predictor.set_image(front_rgb_np)

        masks, scores, logits = self.model_segment_anything_predictor.predict(box=xyxy, multimask_output=True)

        best_score_index = np.argmax(scores)
        best_score_mask = masks[best_score_index]

        if debug:
            self._plot_segment_anything_result(front_rgb_np, best_score_mask)

        # NOT USED: resize mask
        mask_image = Image.fromarray(best_score_mask.astype('uint8') * 255)  # Convert to a PIL image
        resized_mask_image = mask_image.resize((128, 128), Image.NEAREST) # TODO: replace hard-coded image size with dynamic image size from camera
        resized_mask = np.array(resized_mask_image) // 255
        resized_mask = resized_mask.astype(np.bool)

        return resized_mask

    def get_target_object_world_coords(self, front_rgb, points, task_name, debug=True, auto_crop=False):
        # NOTE: modify this when new task is added
        if task_name in ['OpenDrawer', 'open_drawer', 'PutItemInDrawer', 'put_item_in_drawer']:
            text_query = 'drawer frame'
        elif task_name in ['OpenJar', 'open_jar']:
            text_query = 'jar'
        elif task_name in ['HandOverItem', 'hand_over_item']:
            text_query = 'cube'
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!! NotImplementedError in get_target_object_world_coords !!!!!!!!!!!!!!!!!!!!!!!')
            raise NotImplementedError

        bbox = self.get_bounding_box_using_owl_vit(text_query, front_rgb, debug)
        mask = self.get_segmentation_mask_using_segment_anything(bbox, front_rgb, debug)
        self.image_name_counter += 1

        obj_points = points[mask]
        if len(obj_points) == 0:
            raise ValueError(f"Object {text_query} not found in the scene")

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points_downsampled = np.asarray(pcd_downsampled.points)
        target_object_world_coords = np.mean(obj_points_downsampled, axis=0)

        # [NOT USED] find the actual centroid position of the point cloud
        # distances = np.linalg.norm(obj_points_downsampled - target_object_world_coords, axis=1)
        # closest_point_index = np.argmin(distances)
        # target_object_world_coords = obj_points_downsampled[closest_point_index]

        print('target_object_world_coords: ', target_object_world_coords)
        if text_query == 'jar' and (bbox[2] > 0.15 or bbox[3] > 0.15):
            # this can only happens when jar is not detected, and we just pick a center point in the workspace as the crop point
            target_object_world_coords = [0.27462014, -0.00487481, 0.81258505]
            print('!!!! Jar is not detected. Use the center of the workspace. !!!')

        auto_crop_radius = 0.0
        if auto_crop:
            auto_crop_padding = 0.05
            obj_x_min = np.min(obj_points[:, 0])
            obj_x_max = np.max(obj_points[:, 0])
            obj_y_min = np.min(obj_points[:, 1])
            obj_y_max = np.max(obj_points[:, 1])
            obj_z_min = np.min(obj_points[:, 2])
            obj_z_max = np.max(obj_points[:, 2])
            obj_max_dim = np.max([obj_x_max - obj_x_min, obj_y_max - obj_y_min, obj_z_max - obj_z_min])
            auto_crop_radius = obj_max_dim + auto_crop_padding

        return target_object_world_coords, auto_crop_radius

    def reset_image_name_counter(self):
        self.image_name_counter = 0