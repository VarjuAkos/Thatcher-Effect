import cv2
import numpy as np
import torch
from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import get_largest_face
import os
from segment_anything import sam_model_registry, SamPredictor

def extract_feature(image, center, size):
    x, y = center
    w, h = size
    left = max(int(x - w/2), 0)
    top = max(int(y - h/2), 0)
    right = min(int(x + w/2), image.shape[1])
    bottom = min(int(y + h/2), image.shape[0])
    return image[top:bottom, left:right]

def segment_feature(image, sam_predictor):
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Define a box slightly smaller than the image
    box_margin = 0.1  # 10% margin
    box = np.array([
        w * box_margin,  # x1
        h * box_margin,  # y1
        w * (1 - box_margin),  # x2
        h * (1 - box_margin)   # y2
    ])

    # Predict mask
    sam_predictor.set_image(image)
    masks, _, _ = sam_predictor.predict(
        box=box,
        multimask_output=False,
    )

    # Apply mask to image
    mask = masks[0]
    segmented_image = image.copy()
    segmented_image[~mask] = [0, 0, 0]  # Set background to black

    # Draw box on original image for verification
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, 
                  (int(box[0]), int(box[1])), 
                  (int(box[2]), int(box[3])), 
                  (0, 255, 0), 2)

    return segmented_image, image_with_box

def detect_and_export_features(image_path, output_folder, sam_checkpoint):
    # Initialize the face detection model
    det_net = init_detection_model('retinaface_resnet50', half=False)
    
    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    
    # Read the image
    img_ori = cv2.imread(image_path)
    if img_ori is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    h, w = img_ori.shape[:2]
    
    # Detect faces
    with torch.no_grad():
        bboxes = det_net.detect_faces(img_ori, 0.97)
    
    # Get the largest face
    bbox = get_largest_face(bboxes, h, w)[0]
    
    # Extract landmarks
    landmarks = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Draw and export full face with landmarks
    face_with_landmarks = img_ori.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(face_with_landmarks, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(face_with_landmarks, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "face_with_landmarks.jpg"), face_with_landmarks)
    
    # Extract and export eyes (increased size)
    eye_size = (int(w * 0.15), int(h * 0.1))  # Adjust these values to change eye box size
    left_eye = extract_feature(img_ori, landmarks[0], eye_size)
    right_eye = extract_feature(img_ori, landmarks[1], eye_size)
    cv2.imwrite(os.path.join(output_folder, "left_eye.jpg"), left_eye)
    cv2.imwrite(os.path.join(output_folder, "right_eye.jpg"), right_eye)
    
    # Extract and export mouth (increased size)
    mouth_center = ((landmarks[3][0] + landmarks[4][0]) // 2, (landmarks[3][1] + landmarks[4][1]) // 2)
    mouth_size = (int(w * 0.15), int(h * 0.1))  # Adjust these values to change mouth box size
    mouth = extract_feature(img_ori, mouth_center, mouth_size)
    cv2.imwrite(os.path.join(output_folder, "mouth.jpg"), mouth)
    
    # Segment and save features
    segmented_left_eye, left_eye_with_box = segment_feature(left_eye, sam_predictor)
    segmented_right_eye, right_eye_with_box = segment_feature(right_eye, sam_predictor)
    segmented_mouth, mouth_with_box = segment_feature(mouth, sam_predictor)
    
    cv2.imwrite(os.path.join(output_folder, "segmented_left_eye.png"), segmented_left_eye)
    cv2.imwrite(os.path.join(output_folder, "segmented_right_eye.png"), segmented_right_eye)
    cv2.imwrite(os.path.join(output_folder, "segmented_mouth.png"), segmented_mouth)
    
    cv2.imwrite(os.path.join(output_folder, "left_eye_with_box.jpg"), left_eye_with_box)
    cv2.imwrite(os.path.join(output_folder, "right_eye_with_box.jpg"), right_eye_with_box)
    cv2.imwrite(os.path.join(output_folder, "mouth_with_box.jpg"), mouth_with_box)

    print(f"Features exported to {output_folder}")

def main():
    image_path = "1.png"
    output_folder = "output/"
    sam_checkpoint = "../segment-anything/checkpoint/sam_vit_h_4b8939.pth"
    
    try:
        detect_and_export_features(image_path, output_folder, sam_checkpoint)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()