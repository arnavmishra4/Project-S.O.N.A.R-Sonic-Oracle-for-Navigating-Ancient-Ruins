# Cell 2: Sonic Embedding with VGGish

import tensorflow_hub as hub
import numpy as np
import soundfile as sf # Used for efficient chunked reading of audio files
import os
import glob # Needed to find your generated audio files
import json # To load metadata if needed

# --- Configuration for Embedding Module ---
# Path to the directory where your sonified audio files are saved
# This should match your 'output_audio_base_dir' from the previous cell
SONIFIED_AUDIO_BASE_DIR = "/kaggle/input/openai-competition-dataset/sonified_outputs"
EMBEDDING_OUTPUT_DIR = "audio_embeddings" # Directory to save extracted embeddings

os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)

# VGGish model URL from TensorFlow Hub
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"

# VGGish expects audio at 16kHz
VGGISH_SAMPLE_RATE = 16000

print("Cell 2: Sonic Embedding Setup Complete.")

# --- Load the VGGish model ---
# This will download the model weights if not already cached
print(f"Loading VGGish model from: {VGGISH_MODEL_URL}")
try:
    vggish_model = hub.load(VGGISH_MODEL_URL)
    print("VGGish model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load VGGish model. Please check your internet connection or TF Hub installation: {e}")
    # Exit or handle gracefully if the model cannot be loaded
    raise # Raise the exception to halt execution if model load fails

# --- Function to extract VGGish embeddings (Updated for chunked processing) ---
def extract_vggish_embeddings(audio_filepath, target_sample_rate=VGGISH_SAMPLE_RATE, chunk_duration_sec=10):
    """
    Loads an audio file in chunks, resamples each chunk to the target_sample_rate (16kHz for VGGish),
    and extracts VGGish embeddings. This is memory-efficient for large audio files.

    Args:
        audio_filepath (str): Path to the input audio file (.wav).
        target_sample_rate (int): The sample rate expected by VGGish (default 16000 Hz).
        chunk_duration_sec (int): Duration of audio chunks to process at a time (in seconds).

    Returns:
        np.ndarray: A 2D array of VGGish embeddings. Each row is a 128-dimensional embedding
                    for a segment of audio. Returns an empty array if processing fails.
    """
    all_embeddings = []
    
    try:
        with sf.SoundFile(audio_filepath, 'r') as f_read:
            original_sr = f_read.samplerate
            num_channels = f_read.channels
            
            # Determine block size for reading from the original file
            # Aim for blocks of around `chunk_duration_sec` at the original sample rate
            block_size_samples_orig = int(chunk_duration_sec * original_sr)
            
            print(f"    Processing '{os.path.basename(audio_filepath)}' in {chunk_duration_sec}s blocks (original SR: {original_sr} Hz, channels: {num_channels}).")

            # Iterate over audio in blocks
            # `always_2d=True` ensures stereo files give (samples, channels), mono files give (samples, 1)
            for audio_block_orig_sr in f_read.blocks(blocksize=block_size_samples_orig, dtype='float32', always_2d=True):
                # Ensure block is mono. If original is stereo, average across channels.
                if num_channels > 1:
                    audio_block_orig_sr = np.mean(audio_block_orig_sr, axis=1) # Convert to mono (1D array)

                if audio_block_orig_sr.size == 0:
                    continue

                # Resample the current block if necessary
                audio_block_vggish_sr = audio_block_orig_sr
                if original_sr != target_sample_rate:
                    try:
                        import resampy
                        audio_block_vggish_sr = resampy.resample(audio_block_orig_sr, sr_orig=original_sr, sr_new=target_sample_rate)
                    except ImportError:
                        from scipy.signal import resample
                        num_samples_resampled = int(len(audio_block_orig_sr) * (target_sample_rate / original_sr))
                        audio_block_vggish_sr = resample(audio_block_orig_sr, num_samples_resampled)
                    except Exception as e:
                        print(f"    Error during resampling block: {e}. Skipping this block.")
                        continue

                # VGGish model expects float32 input in the range [-1.0, 1.0]
                audio_block_vggish_sr = audio_block_vggish_sr.astype(np.float32)

                # FIXED: Ensure the tensor is 1D (shape=(N,)) as expected by VGGish model
                audio_block_vggish_sr = np.squeeze(audio_block_vggish_sr)
                # If audio_block_vggish_sr was (samples, 1), squeeze makes it (samples,)
                # If it was already (samples,), squeeze does nothing.

                # Only process if the block is long enough for at least one VGGish frame (0.96 sec = 15360 samples at 16kHz)
                min_samples_for_vggish = int(VGGISH_SAMPLE_RATE * 0.96)
                if len(audio_block_vggish_sr) >= min_samples_for_vggish:
                    embeddings_block = vggish_model(audio_block_vggish_sr).numpy()
                    all_embeddings.append(embeddings_block)
                # else:
                #    Optionally print if blocks are too short:
                #    print(f"    Skipping too-short block ({len(audio_block_vggish_sr)} samples) for embedding in {os.path.basename(audio_filepath)}.")


        if all_embeddings:
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"    Extracted {final_embeddings.shape[0]} total embeddings (128-dim each) from '{os.path.basename(audio_filepath)}'.")
            return final_embeddings
        else:
            print(f"    No embeddings extracted from '{os.path.basename(audio_filepath)}'. File might be too short or processing failed.")
            return np.array([])

    except sf.LibsndfileError as e:
        print(f"    Error reading audio file '{audio_filepath}': {e}. This might mean the file is corrupted or not a valid WAV.")
        return np.array([])
    except Exception as e:
        print(f"    An unexpected error occurred during VGGish embedding for '{audio_filepath}': {e}")
        return np.array([])


# --- Main script to process sonified audio files ---
print(f"\n--- Starting VGGish Embedding Extraction from {SONIFIED_AUDIO_BASE_DIR} ---")

for transect_folder in os.listdir(SONIFIED_AUDIO_BASE_DIR):
    transect_path = os.path.join(SONIFIED_AUDIO_BASE_DIR, transect_folder)
    if os.path.isdir(transect_path):
        # Find the full sonification WAV file
        wav_files = glob.glob(os.path.join(transect_path, f"{transect_folder}_full_sonification_SOTA*.wav"))
        
        if not wav_files:
            print(f"No sonified WAV file found for transect '{transect_folder}'. Skipping embedding.")
            continue
        
        # Assuming there's only one relevant WAV file per transect folder
        audio_file_path = wav_files[0]
        print(f"\nProcessing audio for Transect: {transect_folder}")
        print(f"  Input audio file: {audio_file_path}")

        embeddings = extract_vggish_embeddings(audio_file_path)

        if embeddings.size > 0:
            output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, f"{transect_folder}_embeddings.npy")
            np.save(output_filepath, embeddings)
            print(f"  Embeddings saved to: {output_filepath}")
            
            # Optionally, load and inspect the associated geospatial metadata
            metadata_filepath = os.path.join(transect_path, f"{transect_folder}_geospatial_metadata.json")
            if os.path.exists(metadata_filepath):
                with open(metadata_filepath, 'r') as f:
                    cell_metadata = json.load(f)
                
                # Each embedding corresponds to a segment of audio. VGGish segments are 0.96s long.
                # You'll need to align these embeddings back to your original CellGeom objects.
                # This is a crucial step for the anomaly detection module.
                # A simple approximation: each VGGish embedding represents approx 0.96 seconds,
                # with a hop of 0.5 seconds.
                # You'll have N embeddings for an audio of T seconds.
                # num_vggish_embeddings = (T - 0.96) / 0.5 + 1
                # Here, your DURATION_PER_GRID_CELL is 6 seconds.
                # Each 6-second cell would produce (6 - 0.96)/0.5 + 1 = 11.08 -> 11 embeddings
                
                print(f"  Loaded {len(cell_metadata)} geospatial cells and {embeddings.shape[0]} VGGish embeddings.")
                print("  Note: You will need to carefully map VGGish embeddings (0.96s segments) to your original 6-second geospatial cells.")
                # You might store (embedding_index, cell_id) mappings or similar.
                # This could involve creating a new metadata file that links embeddings to their original cell_geom.
            else:
                print(f"  Warning: Geospatial metadata file not found for {transect_folder}. Cannot link embeddings to original cells.")

print("\n--- VGGish Embedding Extraction Complete for all processed transects. ---")
print(f"All embeddings saved to: '{EMBEDDING_OUTPUT_DIR}'")