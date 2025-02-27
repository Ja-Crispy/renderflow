#!/usr/bin/env python
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm
import time
from config import MODELS_ZIP_ID, MODELS_ZIP_SIZE_MB, REQUIRED_FILES

# Add gdown for better Google Drive downloads
try:
    import gdown
except ImportError:
    print("Installing gdown dependency...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive using gdown"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if file already exists
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"✓ Zip file already exists: {output_path}")
        return True
    
    print(f"\n⚠️ Warning: This will download ~{MODELS_ZIP_SIZE_MB/1024:.2f} GB of data.")
    print("Please ensure you have enough disk space and a stable internet connection.")
    
    try:
        input("Press Enter to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        return False
    
    print(f"\nDownloading checkpoints.zip (~{MODELS_ZIP_SIZE_MB/1024:.2f} GB)...")
    print("This may take a while depending on your internet connection.")
    
    try:
        # Use gdown for better Google Drive download handling
        url = f"https://drive.google.com/uc?id={file_id}"
        start_time = time.time()
        
        # Download with gdown (shows its own progress bar)
        success = gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        
        if success:
            download_time = time.time() - start_time
            print(f"✓ Download completed in {download_time/60:.1f} minutes: {output_path}")
            return True
        else:
            print("❌ Download failed")
            return False
        
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        
        print("\nRecommendation: Download the file manually from:")
        print(f"https://drive.google.com/file/d/{file_id}/view?usp=sharing")
        print(f"and place it in this directory as {output_path}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar"""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        # Get total number of files for progress tracking
        with zipfile.ZipFile(zip_path) as zf:
            file_list = zf.namelist()
            total_files = len(file_list)
            
            # Extract with progress bar
            with tqdm(total=total_files, desc="Extracting") as pbar:
                for i, file in enumerate(file_list):
                    zf.extract(file, extract_to)
                    pbar.update(1)
        
        print(f"✓ Extraction completed: {len(file_list)} files")
        return True
        
    except Exception as e:
        print(f"❌ Extraction error: {str(e)}")
        return False

def main():
    """Download and extract model checkpoints"""
    print("RenderFlow Model Downloader")
    print("===========================")
    print("This script will download and extract all required model checkpoints.")
    
    # Define paths
    models_zip = Path("checkpoints.zip")  # Changed to match your online filename
    extract_path = Path(".")
    
    # Create only the outputs directory (not in the zip file but needed by the app)
    Path("outputs").mkdir(exist_ok=True)
    
    # Download zip file
    if not download_from_gdrive(MODELS_ZIP_ID, models_zip):
        return 1
    
    # Extract zip file
    if not extract_zip(models_zip, extract_path):
        return 1
    
    # Optional: Remove zip file after extraction
    if models_zip.exists():
        print("\nCleaning up...")
        try:
            models_zip.unlink()
            print(f"✓ Removed temporary zip file: {models_zip}")
        except Exception as e:
            print(f"⚠️ Could not remove zip file: {str(e)}")
    
    # Verify extraction
    required_files = REQUIRED_FILES
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("\n⚠️ Warning: Some required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease try downloading again or contact the repository maintainer.")
        return 1
    
    print("\n✅ All model checkpoints have been successfully downloaded and extracted!")
    print("    You can now run the RenderFlow application.")
    return 0

if __name__ == "__main__":
    sys.exit(main())