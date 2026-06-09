import numpy as np

def calculate_ear(landmarks, eye_indices):
    """Calculates the Eye Aspect Ratio (EAR) using Euclidean distance."""
    p = [landmarks[i] for i in eye_indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def calculate_mar(landmarks, mouth_indices):
    """Calculates Mouth Aspect Ratio (MAR) across 6 internal control points."""
    p = [landmarks[i] for i in mouth_indices]
    v1 = np.linalg.norm(p[2] - p[3])
    v2 = np.linalg.norm(p[4] - p[5])
    h = np.linalg.norm(p[0] - p[1])
    return (v1 + v2) / (2.0 * h + 1e-6)

def calculate_head_deviation(landmarks):
    """
    Measures both Horizontal Asymmetry (Yaw) and Vertical Compression (Pitch)
    to capture left/right distraction and up/down nodding off.
    """
    # Key structural points: 4=Nose Tip, 33=Left Eye Outer, 263=Right Eye Outer, 152=Chin
    nose_tip = landmarks[4]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[152]
    
    # --- 1. HORIZONTAL TRACKING (YAW) ---
    dist_left = np.abs(nose_tip[0] - left_eye[0])
    dist_right = np.abs(nose_tip[0] - right_eye[0])
    total_span = np.linalg.norm(left_eye - right_eye) + 1e-6
    yaw_deviation = np.abs(dist_left - dist_right) / total_span

    # --- 2. VERTICAL TRACKING (PITCH) ---
    # Midpoint between the eyes
    mid_eyes = (left_eye + right_eye) / 2.0
    
    # Real-time vertical length of the face profile
    current_vertical_span = np.linalg.norm(mid_eyes - chin) + 1e-6
    
    # Distance from nose tip to the eye line along the Y axis
    nose_to_eye_line = np.abs(nose_tip[1] - mid_eyes[1])
    
    # When looking straight, this ratio is stable. 
    # Tilting up or down causes the face to compress vertically, shrinking this ratio.
    pitch_ratio = nose_to_eye_line / current_vertical_span
    
    # We calculate deviation from a standard front-facing baseline ratio (~0.35)
    # If the driver looks too far up or down, this delta increases.
    pitch_deviation = np.abs(0.35 - pitch_ratio)

    # Return the maximum deviation across either axis
    return max(yaw_deviation, pitch_deviation)

