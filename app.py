import cv2
import numpy as np
import torch
from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import get_largest_face
import os
from segment_anything import sam_model_registry, SamPredictor
import json

def extract_feature(image, center, size):
    x, y = center
    w, h = size
    left = max(int(x - w/2), 0)
    top = max(int(y - h/2), 0)
    right = min(int(x + w/2), image.shape[1])
    bottom = min(int(y + h/2), image.shape[0])
    return image[top:bottom, left:right], (left, top, right, bottom)

def segment_feature(image, sam_predictor):
    h, w = image.shape[:2]
    
    box_margin = 0.1  # 10% margin
    box = np.array([
        w * box_margin,
        h * box_margin,
        w * (1 - box_margin),
        h * (1 - box_margin)
    ])

    sam_predictor.set_image(image)
    masks, _, _ = sam_predictor.predict(
        box=box,
        multimask_output=False,
    )

    mask = masks[0]
    
    # Create a transparent background
    segmented_image = np.zeros((h, w, 4), dtype=np.uint8)
    segmented_image[mask] = np.concatenate([image[mask], np.full((mask.sum(), 1), 255, dtype=np.uint8)], axis=1)

    return segmented_image, mask

def segment_face(image_path, output_folder, sam_checkpoint):
    det_net = init_detection_model('retinaface_resnet50', half=False)
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    
    img_ori = cv2.imread(image_path)
    if img_ori is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    h, w = img_ori.shape[:2]
    
    with torch.no_grad():
        bboxes = det_net.detect_faces(img_ori, 0.97)
    
    bbox = get_largest_face(bboxes, h, w)[0]
    
    landmarks = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
    
    os.makedirs(output_folder, exist_ok=True)
    
    eye_size = (int(w * 0.15), int(h * 0.1))
    left_eye, left_eye_box = extract_feature(img_ori, landmarks[0], eye_size)
    right_eye, right_eye_box = extract_feature(img_ori, landmarks[1], eye_size)
    
    mouth_center = ((landmarks[3][0] + landmarks[4][0]) // 2, (landmarks[3][1] + landmarks[4][1]) // 2)
    mouth_size = (int(w * 0.15), int(h * 0.1))
    mouth, mouth_box = extract_feature(img_ori, mouth_center, mouth_size)
    
    segmented_left_eye, left_eye_mask = segment_feature(left_eye, sam_predictor)
    segmented_right_eye, right_eye_mask = segment_feature(right_eye, sam_predictor)
    segmented_mouth, mouth_mask = segment_feature(mouth, sam_predictor)
    
    cv2.imwrite(os.path.join(output_folder, "segmented_left_eye.png"), segmented_left_eye)
    cv2.imwrite(os.path.join(output_folder, "segmented_right_eye.png"), segmented_right_eye)
    cv2.imwrite(os.path.join(output_folder, "segmented_mouth.png"), segmented_mouth)
    
    cv2.imwrite(os.path.join(output_folder, "left_eye_with_box.png"), left_eye_box)
    cv2.imwrite(os.path.join(output_folder, "right_eye_with_box.png"), right_eye_box)
    cv2.imwrite(os.path.join(output_folder, "mouth_with_box.png"), mouth_box)


    # Save bounding boxes and image shape for later use
    data = {
        "image_shape": img_ori.shape[:2],
        "left_eye_box": left_eye_box,
        "right_eye_box": right_eye_box,
        "mouth_box": mouth_box
    }
    with open(os.path.join(output_folder, "feature_data.json"), "w") as f:
        json.dump(data, f)
    
    print(f"Segmentation completed. Results saved in {output_folder}")

def apply_thatcher_effect(image_path, segmentation_folder, output_folder):
    # Load original image
    img_ori = cv2.imread(image_path)
    if img_ori is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    # Load segmented features
    segmented_left_eye = cv2.imread(os.path.join(segmentation_folder, "segmented_left_eye.png"), cv2.IMREAD_UNCHANGED)
    segmented_right_eye = cv2.imread(os.path.join(segmentation_folder, "segmented_right_eye.png"), cv2.IMREAD_UNCHANGED)
    segmented_mouth = cv2.imread(os.path.join(segmentation_folder, "segmented_mouth.png"), cv2.IMREAD_UNCHANGED)
    
    # Load feature data
    with open(os.path.join(segmentation_folder, "feature_data.json"), "r") as f:
        feature_data = json.load(f)
    
    # Increase size of features by 1.5x
    scale_factor = 1.1
    segmented_left_eye = cv2.resize(segmented_left_eye, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    segmented_right_eye = cv2.resize(segmented_right_eye, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    segmented_mouth = cv2.resize(segmented_mouth, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Invert the segmented features
    inverted_left_eye = cv2.flip(segmented_left_eye, -1)
    inverted_right_eye = cv2.flip(segmented_right_eye, -1)
    inverted_mouth = cv2.flip(segmented_mouth, -1)
    
    # Create a copy of the original image for the Thatcher effect
    thatcherized = img_ori.copy()
    
    # Function to overlay a feature on the image
    def overlay_feature(image, feature, box):
        x1, y1, x2, y2 = box
        feature_h, feature_w = feature.shape[:2]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        top = max(center_y - feature_h // 2, 0)
        left = max(center_x - feature_w // 2, 0)
        bottom = min(top + feature_h, image.shape[0])
        right = min(left + feature_w, image.shape[1])
        
        feature_area = feature[:bottom-top, :right-left]
        image_area = image[top:bottom, left:right]
        
        alpha = feature_area[:, :, 3] / 255.0
        for c in range(3):
            image_area[:, :, c] = image_area[:, :, c] * (1 - alpha) + feature_area[:, :, c] * alpha
        
        return image
    
    # Apply inverted features
    thatcherized = overlay_feature(thatcherized, inverted_right_eye, feature_data["left_eye_box"])
    thatcherized = overlay_feature(thatcherized, inverted_left_eye, feature_data["right_eye_box"])
    thatcherized = overlay_feature(thatcherized, inverted_mouth, feature_data["mouth_box"])
    
    # Flip the entire image
    final_thatcherized = cv2.flip(thatcherized, -1)
    
    # Save the result
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "thatcherized.png"), final_thatcherized)
    
    print(f"Thatcher effect applied. Result saved in {output_folder}")

def main():
    image_path = "original.png"
    output_folder = "output/"
    sam_checkpoint = "../segment-anything/checkpoint/sam_vit_h_4b8939.pth"
    segmentation_folder = "segmentation/"
    
    try:
        #segment_face(image_path, segmentation_folder, sam_checkpoint)
        print("Segmentation complete.")
    except Exception as e:
        print(f"An error occurred during segmentation: {str(e)}")
        return
    
    try:
        apply_thatcher_effect(image_path, segmentation_folder, output_folder)
    except Exception as e:
        print(f"An error occurred while applying the Thatcher effect: {str(e)}")

if __name__ == "__main__":
    main()