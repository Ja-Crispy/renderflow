
# Google Drive file ID for the models zip
MODELS_ZIP_ID = "14d4zc2CWGXCjpUVrFDbL9ysYrmsQzu-f"
MODELS_ZIP_SIZE_MB = 6850  # 6.85 GB

# Required files after extraction
REQUIRED_FILES = [
    "checkpoints/latentsync_unet.pt",
    "checkpoints/latentsync_syncnet.pt",
    "checkpoints/whisper/tiny.pt",
    "checkpoints/auxiliary/syncnet_v2.model"
]
