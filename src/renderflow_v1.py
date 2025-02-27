import os
import sys
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import time

# Import Kokoro TTS and soundfile to handle audio synthesis
from kokoro import KPipeline
import soundfile as sf

# Add at the top of the file after imports
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("renderflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("renderflow")

CONFIG_PATH = Path("configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Config path: {CONFIG_PATH.absolute()}")
logger.info(f"Config path exists: {CONFIG_PATH.exists()}")
logger.info(f"Checkpoint path exists: {CHECKPOINT_PATH.exists()}")

# Define permanent directories and ensure they exist
OUTPUTS_DIR = Path("outputs").absolute()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
os.chmod(OUTPUTS_DIR, 0o777)  # Full permissions

# Add after imports
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Config path exists: {CONFIG_PATH.exists()}")
logger.info(f"Checkpoint path exists: {CHECKPOINT_PATH.exists()}")
logger.info(f"Outputs directory exists: {OUTPUTS_DIR.exists()}")

def cleanup_old_files(directory: Path, max_age_hours: int = 24, file_pattern: str = "*"):
    """Clean up files older than max_age_hours in the specified directory."""
    current_time = datetime.now()
    deleted_count = 0
    skipped_count = 0
    
    for file_path in directory.glob(file_pattern):
        if not file_path.is_file():
            continue
            
        file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.total_seconds() > (max_age_hours * 3600):
            try:
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to clean up {file_path}: {e}")
                skipped_count += 1
    
    return {"deleted": deleted_count, "skipped": skipped_count}

# Define the voice options mapping (display name to correct voice code)
voice_options = {
    "American English - Female (Alloy)": "af_alloy",
    "American English - Female (Nova)": "af_nova",
    "American English - Male (Adam)": "am_adam",
    "American English - Male (Eric)": "am_eric",
    "British English - Female (Alice)": "bf_alice",
    "British English - Male (Daniel)": "bm_daniel",
    "Spanish - Female (Dora)": "ef_dora",
    "Spanish - Male (Alex)": "em_alex",
    "French - Female (Siwis)": "ff_siwis",
    "Hindi - Female (Alpha)": "hf_alpha",
    "Hindi - Male (Omega)": "hm_omega",
    "Italian - Female (Sara)": "if_sara",
    "Italian - Male (Nicola)": "im_nicola",
    "Japanese - Female (Alpha)": "jf_alpha",
    "Japanese - Male (Kumo)": "jm_kumo",
    "Portuguese - Female (Dora)": "pf_dora",
    "Portuguese - Male (Alex)": "pm_alex",
    "Chinese - Female (Xiaobei)": "zf_xiaobei",
    "Chinese - Male (Yunjian)": "zm_yunjian"
}

# Voice to language code mapping
voice_to_lang = {
    'af_alloy': 'a', 'af_nova': 'a', 'am_adam': 'a', 'am_eric': 'a',
    'bf_alice': 'b', 'bm_daniel': 'b',
    'ef_dora': 'e', 'em_alex': 'e',
    'ff_siwis': 'f',
    'hf_alpha': 'h', 'hm_omega': 'h',
    'if_sara': 'i', 'im_nicola': 'i',
    'jf_alpha': 'j', 'jm_kumo': 'j',
    'pf_dora': 'p', 'pm_alex': 'p',
    'zf_xiaobei': 'z', 'zm_yunjian': 'z'
}

def process_video(
    video_path,
    audio_path,
    tts_text,
    voice_name,
    guidance_scale,
    inference_steps,
    seed,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Process a video by syncing lip movements to audio.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file (optional)
        tts_text: Text to synthesize into speech (optional)
        voice_name: Voice to use for TTS
        guidance_scale: AI model guidance scale (1.0-3.5)
        inference_steps: Number of inference steps (10-50)
        seed: Random seed for reproducibility
        progress: Gradio progress tracker
    
    Returns:
        tuple: (output_video_path, updated_gallery)
        
    Raises:
        gr.Error: For various processing errors
    """
    start_time = time.time()
    
    try:
        progress(0, desc="Starting process...")
        # Clean up old files first
        cleanup_old_files(OUTPUTS_DIR)

        # Handle video path from Gradio upload
        progress(0.1, desc="Validating input files...")
        if isinstance(video_path, dict) and "name" in video_path:
            video_file_path = Path(video_path["name"])
        else:
            video_file_path = Path(video_path)
            
        if not video_file_path.exists():
            raise gr.Error(f"Video file not found: {video_path}. Please upload a valid video file.")
        video_path = video_file_path.absolute().as_posix()


        # Handle audio path
        if audio_path and isinstance(audio_path, (str, Path)):
            audio_file_path = Path(audio_path)
            if not audio_file_path.exists():
                logger.warning(f"Warning: Audio file not found at {audio_path}, falling back to empty audio")
                audio_path = ""
            else:
                audio_path = audio_file_path.absolute().as_posix()
        else:
            audio_path = ""

        # Replace the TTS processing section
        progress(0.2, desc="Processing audio...")
        if tts_text and tts_text.strip():
            logger.info("Synthesizing audio from text using Kokoro TTS...")
            
            try:
                # Get the voice code from the selected dropdown option
                voice_code = voice_options[voice_name]
                # Map the voice code to the correct language code
                lang_code = voice_to_lang[voice_code]
                
                # Initialize KPipeline with the language code
                logger.info(f"Initializing TTS pipeline with lang_code: {lang_code}")
                pipeline_k = KPipeline(lang_code=lang_code)
                
                # Generate audio with debug info
                logger.info(f"Generating audio for text: '{tts_text}' with voice: {voice_code}")
                audio_generator = pipeline_k(tts_text, voice=voice_code, speed=1, split_pattern=r'\n+')
                
                # Process the generator output
                audio_chunks = []
                for i, (graphemes, phonemes, audio) in enumerate(audio_generator):
                    logger.info(f"Processing chunk {i}:")
                    logger.info(f"Graphemes: {graphemes}")
                    logger.info(f"Phonemes: {phonemes}")
                    if audio is not None:
                        audio_chunks.append(audio)
                
                if not audio_chunks:
                    raise Exception("No audio chunks generated from TTS")
                
                # Concatenate audio chunks if multiple were generated
                import numpy as np
                tts_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
                sample_rate = 24000  # Default sample rate
                
                # Create a unique filename in the outputs directory
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                synthesized_audio_path = OUTPUTS_DIR / f"tts_audio_{current_time}.wav"
                
                # Write out the synthesized audio
                logger.info(f"Saving synthesized audio to: {synthesized_audio_path}")
                sf.write(str(synthesized_audio_path), tts_audio, sample_rate)
                
                # Verify file exists and has size
                if not synthesized_audio_path.exists() or synthesized_audio_path.stat().st_size == 0:
                    raise Exception(f"Failed to create valid audio file at {synthesized_audio_path}")
                
                # Double check file is readable
                try:
                    with open(synthesized_audio_path, 'rb') as f:
                        f.read(1024)  # Try reading first 1KB
                except Exception as e:
                    raise Exception(f"Audio file created but not readable: {e}")
                
                # Update audio_path with verified file
                audio_path = str(synthesized_audio_path.absolute())
                logger.info(f"Audio file created and verified at: {audio_path}")
                
                # Sleep briefly to ensure file is fully written
                time.sleep(1)
                progress(0.3, desc="Generating speech...")
                progress(0.4, desc="Saving audio file...")

            except Exception as tts_error:
                logger.error(f"TTS Error: {tts_error}")
                import traceback
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
                raise gr.Error(f"Kokoro TTS failed: {tts_error}")

        # Set the output path for the processed video
        output_path = str(OUTPUTS_DIR / f"{video_file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

        # Load and validate config
        progress(0.5, desc="Loading model configuration...")
        try:
            config = OmegaConf.load(CONFIG_PATH)
            config["run"].update({
                "guidance_scale": guidance_scale,
                "inference_steps": inference_steps,
            })
        except Exception as e:
            raise gr.Error(f"Failed to load config: {str(e)}")

        # Parse the arguments
        args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

        logger.info(f"Processing video with parameters:")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Audio path: {audio_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Guidance scale: {guidance_scale}")
        logger.info(f"Inference steps: {inference_steps}")
        logger.info(f"Seed: {seed}")

        progress(0.6, desc="Processing video...")
        _ = main(
            config=config,
            args=args,
        )
        
        progress(0.9, desc="Finalizing...")
        # Verify output was created
        if not Path(output_path).exists():
            raise gr.Error("Processing completed but output file was not created")
            
        progress(1.0, desc="Processing completed successfully!")
        logger.info("Processing completed successfully.")
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Return with info for info box
        info_text = f"Processing completed in {processing_time:.2f} seconds\n"
        info_text += f"Video: {video_file_path.name}\n"
        info_text += f"Parameters: Steps={inference_steps}, Guidance={guidance_scale}, Seed={seed}"
        
        return [output_path, list_processed_videos(), info_text]

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error after {processing_time:.2f} seconds: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")

def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    """
    Create argument namespace for the inference script.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        output_path: Path where processed video will be saved
        inference_steps: Number of inference steps to run
        guidance_scale: Scale for guidance (higher = stronger effect)
        seed: Random seed for reproducibility
        
    Returns:
        argparse.Namespace: Arguments for inference script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args([
        "--inference_ckpt_path", str(CHECKPOINT_PATH.absolute()),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
    ])

# Function to list processed videos in the outputs directory# Function to generate video cards with playable previews
def list_processed_videos():
    """
    Get list of processed videos for Gallery component.
    
    Returns:
        list: List of tuples containing (video_path, caption)
    """
    video_files = []
    
    # Get all MP4 files in the output directory, sorted by modification time (newest first)
    for file_path in sorted(OUTPUTS_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        # Create caption with file name and date
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        file_size = f"{file_path.stat().st_size / (1024 * 1024):.1f} MB"
        caption = f"{file_path.stem}\n{file_time} • {file_size}"
        
        # Add tuple of (file_path, caption)
        video_files.append((str(file_path.absolute()), caption))
    
    return video_files

# Create Gradio interface with the Soft theme
with gr.Blocks(theme=gr.themes.Soft(), title="RenderFlow v1") as demo:
    gr.Markdown("""
    # RenderFlow v1 🎥✨
    **Bring your videos to life with AI-powered lip-syncing and voice synthesis!**
    """)

    with gr.Tabs():
        # Tab 1: Process Video
        with gr.Tab("Process Video 🎬"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload your video and audio or generate speech from text!")
                    video_input = gr.Video(label="🎥 Upload Video")
                    audio_input = gr.Audio(label="🎵 Upload Audio (optional)", type="filepath")
                    tts_text = gr.Textbox(
                        label="📝 Enter Text for TTS",
                        placeholder="Type your text here to generate speech (overrides audio file)",
                        lines=3
                    )
                    voice_dropdown = gr.Dropdown(
                        choices=list(voice_options.keys()),
                        label="🗣️ Choose Voice for TTS",
                        value="American English - Female (Nova)"
                    )
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=3.5,
                            value=1.5,
                            step=0.5,
                            label="🎚️ Guidance Scale"
                        )
                        inference_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=1,
                            label="⏱️ Inference Steps"
                        )
                    with gr.Row():
                        seed = gr.Number(value=1247, label="🎲 Random Seed", precision=0)
                    
                    # Add processing status
                    status = gr.Markdown("### Status: Ready")
                    
                    process_btn = gr.Button("🚀 Process Video", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### Output Video")
                    video_output = gr.Video(label="🎥 Output Video")
                    info_box = gr.Textbox(label="ℹ️ Processing Info", interactive=False)

        # Tab 2: Gallery
        with gr.Tab("Gallery 🖼️"):
            gr.Markdown("### Processed Videos")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
            
            # Use Gallery component instead of HTML
            gallery = gr.Gallery(
                value=list_processed_videos,
                columns=3,
                rows=2,
                object_fit="contain",
                height="auto",
                label="Processed Videos",
                show_label=False,
                elem_id="video_gallery",
                allow_preview=True,
                preview=True
            )
            
            # Add gallery refresh functionality
            refresh_btn.click(
                fn=list_processed_videos,
                outputs=gallery
            )

        def on_process_success():
            """Callback for successful processing"""
            return "### Status: Processing completed successfully! ✅"

        def on_process_error():
            """Callback for processing errors"""
            return "### Status: Error occurred during processing ❌"

        process_event = process_btn.click(
            fn=process_video,
            inputs=[
                video_input, 
                audio_input, 
                tts_text, 
                voice_dropdown, 
                guidance_scale, 
                inference_steps, 
                seed
            ],
            outputs=[video_output, gallery, info_box]
        )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)