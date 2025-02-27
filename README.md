# RenderFlow

RenderFlow is a powerful video processing and rendering application that utilizes state-of-the-art AI models for various video manipulation tasks.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (CUDA 12.1)
- At least 12GB of free disk space for model checkpoints
- Windows/Linux operating system

## Quick Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download model checkpoints (choose one method):

   **Option 1: Automatic Download**
   ```bash
   python download_models.py
   ```

   **Option 2: Manual Download**
- Download [`checkpoints.zip` from the provided Google Drive link](https://drive.google.com/file/d/14d4zc2CWGXCjpUVrFDbL9ysYrmsQzu-f/view?usp=sharing)
   - Place the downloaded file in the project root directory
   - Extract the contents of the zip file in the project root directory

4. Run the application:
   ```bash
   python src/renderflow_v1.py
   ```

## Project Structure

```
renderflow/
├── checkpoints/     # Model checkpoint files
├── configs/         # Configuration files
├── latentsync/      # Latent synchronization module
├── outputs/         # Generated output files
├── scripts/         # Utility scripts
├── src/            # Source code
├── temp/           # Temporary files
├── config.py       # Configuration settings
├── download_models.py  # Model downloader script
└── requirements.txt    # Python dependencies
```

## Features

## Features

- **AI-Powered Video Enhancement**: Improve video quality using advanced AI models for upscaling, denoising, and color correction.
- **Real-Time Processing**: Leverage GPU acceleration for real-time video processing and rendering.
- **Flexible Input Formats**: Support for various video formats including MP4, AVI, MOV, and more.
- **Customizable Pipelines**: Create and customize video processing pipelines to suit your specific needs.
- **Batch Processing**: Process multiple videos simultaneously to save time and increase productivity.
- **User-Friendly Interface**: Intuitive GUI for easy navigation and operation.
- **Extensive Documentation**: Comprehensive guides and documentation to help you get started quickly.


## Notes

- The model checkpoints are large files (~7GB) and require a stable internet connection to download
- Make sure you have enough disk space before downloading the checkpoints
- The application requires a CUDA-compatible GPU for optimal performance

## Troubleshooting

If you encounter any issues during setup:

1. Ensure your Python version is compatible
2. Verify that all required dependencies are installed correctly
3. Check if the model checkpoints are downloaded and extracted properly
4. Make sure your GPU drivers are up to date

