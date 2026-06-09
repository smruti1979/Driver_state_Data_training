# Facial Landmark Indices from MediaPipe Face Mesh Map
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Tracks the actual air gap across 6 explicit inner-mouth points
MOUTH = [78, 308, 13, 14, 312, 317] 

# Detection Thresholds (Fine-Tuned for front-facing camera)
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.55          # ADJUSTED: Sane baseline for the 6-point inner gap math
HEAD_TILT_THRESHOLD = 0.25

# Time Thresholds (Consecutive frames at 30 FPS)
EYE_CLOSED_FRAMES = 30  # ~1.1 seconds (Ignores quick blinks)
YAWN_FRAMES = 40        # ~0.8 seconds (Captures yawning quickly)
DISTRACTED_FRAMES = 12  # ~0.5 seconds (Catches rapid head drops)

# Audio Asset Configuration
AUDIO_WARNING = "media/warning.wav"
AUDIO_CRITICAL = "media/critical.wav"

# UI Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)

# Storage
LOG_FILE = "telemetry_log.csv"
