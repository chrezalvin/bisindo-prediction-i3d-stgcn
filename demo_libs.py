# import libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import random
import math
import keras as kr
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.ops import mean
from custom_stgcn import Model as STGCNModel
from keras import Model
from keras.layers import Dropout, Reshape, Lambda, Activation

def forceFrames(frames: list, target_length=64) -> list:
    """
    Preprocess video by adding a padding if the number of frames is less than target_length. And resize each frame to target_size.
    """
    frame_count = len(frames)
    if frame_count < target_length:
        # Randomly choose to pad with first or last frame
        randInt = random.randint(0, 1) # if 1, pad with last frame, else pad with first frame
        pad_frame = frames[0] if randInt == 0 else frames[-1]
        # Create padding frames
        padding = [pad_frame] * (target_length - frame_count)
        if randInt == 0:
            frames = padding + frames
        else:
            frames = frames + padding
    elif frame_count > target_length:
        # Randomly select a starting index to cut the video
        start_idx = np.random.randint(0, frame_count - target_length + 1)
        frames = frames[start_idx : start_idx + target_length]

    return frames

def loadFrameFromVideoPath(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = cap.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)        

    cap.release()

    return frames

def preprocess_video(video_path, yolo_person_estimation_model, area_factor=math.sqrt(2)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}")
    
    frames = loadFrameFromVideoPath(video_path)
    
    predictions = yolo_person_estimation_model.predict(source=frames, save=False, verbose=False, stream=False)

    cropped_frames = []
    prev_pred = None
    for idx, prediction in enumerate(predictions):
        frame = frames[idx]
        if len(prediction.boxes) == 0:
            # uses the previous prediction if no person is detected
            if prev_pred is None:
                raise ValueError("No person detected in the video frames.")
            center_x, center_y, width, height = prev_pred
        else:
            center_x, center_y, width, height = prediction.boxes.xywh[0]

        linear_factor = math.sqrt(area_factor)  # Linear scaling (√(√2) ≈ 1.189)

        # Apply linear scaling factor
        new_width = width * linear_factor
        new_height = height * linear_factor

        e_x1 = center_x - new_width / 2
        e_y1 = center_y - new_height / 2
        e_x2 = center_x + new_width / 2
        e_y2 = center_y + new_height / 2

        # Clamp to image dimensions
        h, w = frame.shape[:2]
        e_x1 = max(0, e_x1)
        e_y1 = max(0, e_y1)
        e_x2 = min(w, e_x2)
        e_y2 = min(h, e_y2)

        # crop the bounding box
        cropped_frame = frame[int(e_y1):int(e_y2), int(e_x1):int(e_x2)]
        # resize to 256x256
        cropped_frame = cv2.resize(cropped_frame, (256, 256))
        cropped_frames.append(cropped_frame)
        prev_pred = (center_x, center_y, width, height)

    return cropped_frames

def find_path_from_id(directory, id):
    """
    Finds the path of the file with the given id in the dataset directory.
    the folder have depths so it will search recursively
    extension can be .mov or .mp4
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith(id) and (file.endswith(".mov") or file.endswith(".mp4")):
                return os.path.join(root, file)
    return None

def calculateOpticalFlowFarneback(prev_frame, frame):
    # convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def calculateOpticalFlowTvl1(frames: list):
    """Calculate optical flow using the TV-L1 method."""
    # Initialize the TV-L1 optical flow object
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    flow = []

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        # Calculate optical flow
        flow.append(tvl1.calc(prev_gray, gray, None))
        prev_gray = gray

    return flow

# get optical flow using farneback method
def getOpticalFlow(frames: list):
    # add extra frame at the end to match the length
    frames.append(frames[-1])
    optical_flow = []
    prev_gray = None

    for frame in frames:
        if prev_gray is None:
            prev_gray = frame
            continue
        
        flow = calculateOpticalFlowFarneback(prev_gray, frame)
        optical_flow.append(flow)
        
        prev_gray = frame

    return optical_flow

# function to visualize optical flow
def visualize_flow_hsv(flow, actual_frame):
    """Convert flow to HSV color space for intuitive visualization."""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Max saturation
    
    # Convert flow to polar coordinates (angle and magnitude)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction (0-180°)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = speed
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plt.subplot(1, 2, 1)
    plt.imshow(actual_frame)  # Show the first frame for reference
    plt.title('Original Frame')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(rgb)
    plt.title('Optical Flow (HSV Color Coding)')
    plt.axis('off')
    plt.show()

# only includes left/right shoulder and elbow
body_pose_lookup = [14, 12, 11, 13]
body_pose_joints = [[14, 12], [12, 11], [11, 13]]

# uses all hand poses
hand_pose_joints = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17]
]

def predict_pose_and_hand_keypoints(results_pose: list, results_hand: list):
    """
    Predict the pose and hand keypoints from the MediaPipe result.
    Returns a dictionary with pose keypoints and hand keypoints.
    """
    pose_keypoints = []
    left_hand_keypoints = []
    right_hand_keypoints = []

    for result in results_hand:
        left_hand = np.zeros((21, 3), dtype=np.float16)
        right_hand = np.zeros((21, 3), dtype=np.float16)

        for idx, handedness in enumerate(result.handedness):
            list_hand = []

            for landmark in result.hand_landmarks[idx]:
                list_hand.append((landmark.x, landmark.y, landmark.z))
            
            if handedness[0].index == 0:  # left hand
                left_hand = np.array(list_hand, dtype=np.float16)
            if handedness[0].index == 1: # right hand
                right_hand = np.array(list_hand, dtype=np.float16)
        
        left_hand_keypoints.append(left_hand)
        right_hand_keypoints.append(right_hand)

    # Extract pose keypoints
    for result in results_pose:
        pose = []  # 4 joints, 3 coordinates (x, y, z)
        # check if pose_landmarks is not empty
        if not result.pose_landmarks:
            # if no pose landmarks detected, append zeroes
            pose = np.zeros((4, 3), dtype=np.float16)
        else:
            for idx, landmark in enumerate(result.pose_landmarks[0]):
                if idx not in body_pose_lookup:
                    continue
                pose.append((landmark.x, landmark.y, landmark.z))
    
        pose_keypoints.append(pose)

    return {
        "pose": np.array(pose_keypoints, dtype=np.float16),
        "left_hand": np.array(left_hand_keypoints, dtype=np.float16),
        "right_hand": np.array(right_hand_keypoints, dtype=np.float16)
    }

def pred_mp_frames(pose_predictor, hand_predictor, frames: list):
    pose_preds = []
    hand_preds = []
    for frame in frames:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_preds.append(hand_predictor.detect(mp_image))
        pose_preds.append(pose_predictor.detect(mp_image))

    pose_pred_res = predict_pose_and_hand_keypoints(pose_preds, hand_preds)
    return pose_pred_res

def baseline_model_i3d(
        num_classes: int, 
        learning_rate: float = 1e-3, 
        weight_decay: float = 1e-7
        ) -> Model:
    base_model = Inception_Inflated3d(
        include_top=False,
        weights="rgb_imagenet_and_kinetics",
        input_shape=(64, 224, 224, 3),
    )

    # turn on training for the base model
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output

    x = Dropout(0.5, name='Dropout_5a')(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames = int(x.shape[1])  # Number of frames in the input video
    x = Reshape((num_frames, num_classes))(x)

    x = Lambda(lambda x: mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)

    model_i3d = Model(inputs=base_model.input, outputs=x, name='Inception_Inflated3d')

    model_i3d.compile(
        optimizer=kr.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=kr.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model_i3d

def baseline_model_i3d_flow(
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-7
) -> Model:
    base_model = Inception_Inflated3d(
        include_top=False,
        weights="flow_kinetics_only",
        input_shape=(64, 224, 224, 2),
    )

    x = base_model.output

    x = Dropout(0.5, name='Dropout_5a')(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames = int(x.shape[1])  # Number of frames in the input video
    x = Reshape((num_frames, num_classes))(x)

    x = Lambda(lambda x: mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)

    model_flow = Model(inputs=base_model.input, outputs=x, name='Inception_Inflated3d')

    model_flow.compile(
        optimizer=kr.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=kr.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model_flow

def baseline_model_stgcn(
    num_classes: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-7
):
    model_pose = STGCNModel(num_classes=num_classes)

    dummy_input = np.random.normal(size=(1, 3, 64, 46, 1))  # Note: 3 channels (x,y,z coordinates)
    _ = model_pose(dummy_input, training=True)  # This builds the model

    model_pose.compile(
        optimizer=kr.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=kr.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model_pose


def mirror_video(frames: list):
    """
    Mirrors the video frames horizontally
    """
    return [cv2.flip(frame, 1) for frame in frames]

def randomCropping(frames: list, crop_size=(224, 224)):
    """
    Randomly crops the video frames to the specified size
    """
    # Check if the frames are longer than the crop size
    h, w = frames[0].shape[:2]
    ch, cw = crop_size

    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than frame dimensions")

    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)

    cropped_frames = [frame[top:top + ch, left:left + cw] for frame in frames]
    
    return cropped_frames

def i3d_normalization(x: np.ndarray) -> np.ndarray:
    """
    Normalize input frames for I3D model.
    The normalization is done by scaling pixel values to the range [-1, 1].
    """
    x = x.astype(np.float16)
    x = (x / 127.5) - 1.0
    return x

def forceFrames(frames: list, num_frames=64, padding_technique="firstOrLast", chance=0.5):
    """
    Forces the video frames to have a fixed number of frames.
    assuming the frames have less length than num_frames frames.
    """
    # turn into list of np.array
    frames = [np.array(frame) for frame in frames]
    random_choice = np.random.rand()
    if padding_technique == "firstOrLast":
        # pad with first or last frame
        if random_choice > chance:
            # pad with first element at the start
            padding_frame = frames[0]
            data = [padding_frame] * (num_frames - len(frames)) + frames
        else:
            # pad with last element at the end
            padding_frame = frames[-1]
            data = frames + [padding_frame] * (num_frames - len(frames))

    elif padding_technique == "zero":
        # pad with zero frames first or last
        single_frame_shape = frames[0].shape
        padding_frame = np.zeros(single_frame_shape, dtype=frames[0].dtype)
        
        if random_choice > chance:
            # pad with zero frame at the start
            data = [padding_frame] * (num_frames - len(frames)) + frames
        else:
            # pad with zero frame at the end
            data = frames + [padding_frame] * (num_frames - len(frames))

    return data

def temporalCropping(frames: list, num_frames=64):
    """
    Do a temporal cropping of the video frames to force the number of frames to be num_frames.
    get a random start index and crop the frames to the specified number of frames.
    assuming the frames have more length than num_frames frames.
    """
    start_index = np.random.randint(0, len(frames) - num_frames + 1)
    cropped_frames = frames[start_index:start_index + num_frames]
    return cropped_frames


def combinePose(pose_dict):
    """
    Combine pose body, left hand, and right hand into a single array.
    """
    combined = []
    # each frame
    frame_count = len(pose_dict['pose'])
    for frame in range(frame_count):
        temp = []
        for key in ['pose', 'left_hand', 'right_hand']:
            for idx in range(len(pose_dict[key][frame])):
                temp.append(pose_dict[key][frame][idx])
        combined.append(temp)
    return np.array(combined, dtype=np.float16)

def augment_i3d(
        frames: list, 
        dim=(64, 224, 224, 3),
        normalize=True,
        padding_technique="firstOrLast",
    ):
    X = np.empty((1, *dim), dtype=np.float16)

    if len(frames) < dim[0]:
        # if the video has less than 64 frames, pad it
        frames = forceFrames(frames, num_frames=dim[0], padding_technique=padding_technique)
    elif len(frames) > dim[0]:
        # if the video has more than 64 frames, crop it
        frames = temporalCropping(frames, num_frames=dim[0])

    # do a random cropping
    frames = randomCropping(frames, crop_size=(224, 224))

    # 50% chance to mirror the video frames
    if np.random.rand() > 0.5:
        frames = mirror_video(frames)

    # Convert to numpy array and normalize
    frames = np.array(frames, dtype=np.float16)

    if normalize:
        X[0,] = i3d_normalization(frames)
    else:
        X[0,] = frames

    return X

def augment_stgcn(
        poseDict: dict,
        dim=(64, 46, 3),
        padding_technique="firstOrLast"
    ):
        X = np.empty((1, dim[2], dim[0], dim[1], 1), dtype=np.float16)

        poseDict = combinePose(poseDict)

        if len(poseDict) < dim[0]:
            # if the video has less than 64 frames, pad it
            poseDict = forceFrames(poseDict, num_frames=dim[0], padding_technique=padding_technique)
        elif len(poseDict) > dim[0]:
            poseDict = temporalCropping(poseDict, num_frames=dim[0])

        # add tensor dimension
        poseDict = np.expand_dims(poseDict, axis=-1)
        poseDict = np.transpose(poseDict, (2, 0, 1, 3))  

        # Convert to numpy array
        poseDict = np.array(poseDict, dtype=np.float16)

        X[0,] = poseDict

        return X