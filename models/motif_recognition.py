# Cell 4: Archaeological Signature Recognition with DTW

import numpy as np
import os
import glob
import json
from fastdtw import fastdtw # For efficient Dynamic Time Warping
from scipy.spatial.distance import euclidean # For DTW distance metric
from config import EMBEDDING_OUTPUT_DIR, ANOMALY_OUTPUT_DIR, MOTIF_OUTPUT_DIR

EMBEDDING_INPUT_DIR = EMBEDDING_OUTPUT_DIR
ANOMALY_RESULTS_DIR = ANOMALY_OUTPUT_DIR

os.makedirs(MOTIF_OUTPUT_DIR, exist_ok=True)

# Use from config or define as before
ARCHAEOLOGICAL_MOTIF_TRANSECTS = TRANSFORMER_TRANSECTS
 # From your Cell 1 configuration

# Define a placeholder for motifs and their types
# In a real scenario, you would manually define precise time segments
# within your archaeological transects that correspond to specific motifs.
# For this example, we'll simulate by taking a "representative" slice.
# You would replace these with actual start_ms, end_ms from your known sites.
ARCHAEOLOGICAL_MOTIFS_DEFINITIONS = {
    "BR_AC_10": {
        "motif_type": "Geoglyph_Cluster",
        "audio_segment_start_ms": 30000, # Example: 30-40 seconds into the full sonification
        "audio_segment_end_ms": 40000
    },
    "BR_RO_05": {
        "motif_type": "Earthwork_Corridor",
        "audio_segment_start_ms": 60000, # Example: 60-70 seconds
        "audio_segment_end_ms": 70000
    },
    "BR_PA_02": {
        "motif_type": "Xingu_Settlement",
        "audio_segment_start_ms": 150000, # Example: 150-160 seconds
        "audio_segment_end_ms": 160000
    },
    # Add more archaeological sites and their motif definitions here
    # Based on your anomaly regions defined in Cell 1 for ANOMALY_TRIGGERED:
    "BR_AC_07": {
        "motif_type": "Circular_Earthwork",
        "audio_segment_start_ms": 25000, # Example, adjust based on actual anomaly location
        "audio_segment_end_ms": 35000
    },
    "BR_AC_09": {
        "motif_type": "Rectangular_Structure",
        "audio_segment_start_ms": 40000, # Example, adjust based on actual anomaly location
        "audio_segment_end_ms": 50000
    },
}

# Threshold for DTW similarity (lower means more similar)
# This will require experimentation. Anomalies with DTW distance below this
# might be considered a "match".
DTW_SIMILARITY_THRESHOLD = 75 # This is a placeholder, tune based on your data


print("Cell 4: Archaeological Signature Recognition Setup Complete.")

# --- Helper function to get VGGish embeddings for a specific audio time range ---
def get_vggish_embeddings_for_time_range(embeddings_array, audio_start_ms, audio_end_ms, total_audio_duration_ms):
    """
    Retrieves VGGish embeddings corresponding to a specific audio time range.

    Args:
        embeddings_array (np.ndarray): The full array of VGGish embeddings for the transect.
        audio_start_ms (int): Start time of the segment in milliseconds.
        audio_end_ms (int): End time of the segment in milliseconds.
        total_audio_duration_ms (int): Total duration of the full sonified audio in milliseconds.

    Returns:
        np.ndarray: VGGish embeddings for the specified time range.
    """
    if embeddings_array.size == 0 or total_audio_duration_ms == 0:
        return np.array([])

    vggish_frame_length_s = 0.96
    vggish_hop_length_s = 0.5
    
    # Calculate the start and end VGGish frame indices
    # This is an approximation due to overlapping frames.
    # A more precise method would involve aligning VGGish timestamps.
    start_s = audio_start_ms / 1000.0
    end_s = audio_end_ms / 1000.0

    # Ensure indices are within bounds
    num_total_embeddings = embeddings_array.shape[0]

    # Approximate index calculation. VGGish frames are centered.
    # An embedding at index `i` is roughly centered at `i * vggish_hop_length_s + vggish_frame_length_s / 2`
    # This needs careful alignment for production, but this will work for a first pass.
    start_embedding_idx = max(0, int((start_s - vggish_frame_length_s / 2) / vggish_hop_length_s))
    end_embedding_idx = min(num_total_embeddings, int((end_s - vggish_frame_length_s / 2) / vggish_hop_length_s) + 1) # +1 to include the last partial frame

    # If the time range is very short or at the very end, ensure at least one embedding is captured if possible
    if start_embedding_idx >= num_total_embeddings:
        return np.array([])
    if end_embedding_idx <= start_embedding_idx and start_embedding_idx < num_total_embeddings:
        end_embedding_idx = start_embedding_idx + 1 # At least one embedding if within bounds

    return embeddings_array[start_embedding_idx:end_embedding_idx]


# --- 1. Build Archaeological Motif Library ---
print("\n--- Building Archaeological Motif Library ---")
motif_library = {} # Stores {motif_type: [list_of_motif_embeddings]}

for transect_id, motif_info in ARCHAEOLOGICAL_MOTIFS_DEFINITIONS.items():
    motif_type = motif_info["motif_type"]
    audio_start_ms = motif_info["audio_segment_start_ms"]
    audio_end_ms = motif_info["audio_segment_end_ms"]

    full_embedding_filepath = os.path.join(EMBEDDING_INPUT_DIR, f"{transect_id}_embeddings.npy")
    full_metadata_filepath = os.path.join(SONIFIED_AUDIO_BASE_DIR, transect_id, f"{transect_id}_geospatial_metadata.json")
    
    if not os.path.exists(full_embedding_filepath) or not os.path.exists(full_metadata_filepath):
        print(f"  Warning: Skipping motif '{motif_type}' from {transect_id}. Required files not found.")
        continue

    transect_embeddings = np.load(full_embedding_filepath)
    
    with open(full_metadata_filepath, 'r') as f:
        cell_geometries = json.load(f)
    
    # Calculate total audio duration from the metadata
    if cell_geometries:
        total_audio_duration_ms = cell_geometries[-1]['audio_end_ms'] # Last cell's end time is total duration
    else:
        total_audio_duration_ms = 0

    motif_embeddings = get_vggish_embeddings_for_time_range(
        transect_embeddings, audio_start_ms, audio_end_ms, total_audio_duration_ms
    )

    if motif_embeddings.size > 0:
        if motif_type not in motif_library:
            motif_library[motif_type] = []
        motif_library[motif_type].append(motif_embeddings)
        print(f"  Added motif '{motif_type}' from {transect_id} ({motif_embeddings.shape[0]} embeddings).")
    else:
        print(f"  Warning: No VGGish embeddings found for specified motif time range in {transect_id} for '{motif_type}'.")

if not motif_library:
    print("ERROR: No archaeological motifs could be loaded. Cannot proceed with signature recognition.")
    exit()

# --- 2. Iterate Through Anomalies and Perform Motif Matching ---
print("\n--- Performing Archaeological Signature Recognition (Motif Matching) ---")

all_transect_motif_results = {}

# Iterate over the anomaly results saved in the previous step
# You will need to load the anomaly_results.json for each transect
for transect_id in TRANSECTS_TO_ANALYZE:
    anomaly_results_filepath = os.path.join(ANOMALY_RESULTS_DIR, f"{transect_id}_anomaly_results.json")
    full_embedding_filepath = os.path.join(EMBEDDING_INPUT_DIR, f"{transect_id}_embeddings.npy")
    full_metadata_filepath = os.path.join(SONIFIED_AUDIO_BASE_DIR, transect_id, f"{transect_id}_geospatial_metadata.json")

    if not os.path.exists(anomaly_results_filepath) or \
       not os.path.exists(full_embedding_filepath) or \
       not os.path.exists(full_metadata_filepath):
        print(f"  Skipping motif recognition for {transect_id}: Missing anomaly results, embeddings, or metadata.")
        continue

    with open(anomaly_results_filepath, 'r') as f:
        anomaly_data_for_transect = json.load(f)
    
    transect_embeddings = np.load(full_embedding_filepath)
    with open(full_metadata_filepath, 'r') as f:
        cell_geometries = json.load(f)
    
    total_audio_duration_ms = cell_geometries[-1]['audio_end_ms'] if cell_geometries else 0

    motif_matching_results_for_transect = []

    for anomaly_cell_info in anomaly_data_for_transect:
        if anomaly_cell_info["is_anomalous_flag"]:
            cell_minx = anomaly_cell_info["minx"]
            cell_miny = anomaly_cell_info["miny"]
            cell_maxx = anomaly_cell_info["maxx"]
            cell_maxy = anomaly_cell_info["maxy"]
            cell_audio_start_ms = anomaly_cell_info["audio_start_ms"]
            cell_audio_end_ms = anomaly_cell_info["audio_end_ms"]

            # Get the VGGish embeddings for this anomalous cell's audio segment
            anomaly_segment_embeddings = get_vggish_embeddings_for_time_range(
                transect_embeddings, cell_audio_start_ms, cell_audio_end_ms, total_audio_duration_ms
            )

            best_match_motif_type = "Unknown"
            best_match_dtw_distance = float('inf')

            if anomaly_segment_embeddings.size > 0:
                for motif_type, motif_embedding_list in motif_library.items():
                    for single_motif_embeddings in motif_embedding_list:
                        if single_motif_embeddings.size == 0:
                            continue

                        # Perform DTW comparison
                        # Make sure both sequences have enough points for DTW
                        if anomaly_segment_embeddings.shape[0] > 1 and single_motif_embeddings.shape[0] > 1:
                            distance, path = fastdtw(anomaly_segment_embeddings, single_motif_embeddings, dist=euclidean)
                        elif anomaly_segment_embeddings.shape[0] == 1 and single_motif_embeddings.shape[0] == 1:
                            # If both are single embeddings, just use Euclidean distance
                            distance = euclidean(anomaly_segment_embeddings[0], single_motif_embeddings[0])
                        else:
                            # Handle cases where one or both are too short for DTW
                            # You might set a very high distance or skip
                            distance = float('inf') # Set to infinity if comparison isn't meaningful

                        if distance < best_match_dtw_distance:
                            best_match_dtw_distance = distance
                            best_match_motif_type = motif_type
            
            # Decide if it's a "match" based on a threshold
            is_matched = best_match_dtw_distance < DTW_SIMILARITY_THRESHOLD

            motif_matching_results_for_transect.append({
                "cell_id": anomaly_cell_info["cell_id"],
                "minx": cell_minx,
                "miny": cell_miny,
                "maxx": cell_maxx,
                "maxy": cell_maxy,
                "audio_start_ms": cell_audio_start_ms,
                "audio_end_ms": cell_audio_end_ms,
                "is_anomalous_flag": True, # It's an anomalous cell from previous stage
                "mean_anomaly_score": anomaly_cell_info["mean_anomaly_score"],
                "matched_motif_type": best_match_motif_type if is_matched else "No_Match",
                "motif_similarity_score": float(best_match_dtw_distance) if best_match_dtw_distance != float('inf') else None,
                "is_motif_matched": is_matched
            })
        else:
            # Include non-anomalous cells in the output for completeness, but without motif matching
            motif_matching_results_for_transect.append({
                "cell_id": anomaly_cell_info["cell_id"],
                "minx": anomaly_cell_info["minx"],
                "miny": anomaly_cell_info["miny"],
                "maxx": anomaly_cell_info["maxx"],
                "maxy": anomaly_cell_info["maxy"],
                "audio_start_ms": anomaly_cell_info["audio_start_ms"],
                "audio_end_ms": anomaly_cell_info["audio_end_ms"],
                "is_anomalous_flag": False,
                "mean_anomaly_score": anomaly_cell_info["mean_anomaly_score"],
                "matched_motif_type": "Not_Anomalous",
                "motif_similarity_score": None,
                "is_motif_matched": False
            })

    all_transect_motif_results[transect_id] = motif_matching_results_for_transect

    # Save the motif recognition results for the current transect
    output_json_filepath = os.path.join(MOTIF_OUTPUT_DIR, f"{transect_id}_motif_recognition_results.json")
    with open(output_json_filepath, 'w') as f:
        json.dump(motif_matching_results_for_transect, f, indent=4)
    print(f"  Motif recognition results saved to: {output_json_filepath}")

print("\n--- Archaeological Signature Recognition Process Complete ---")
print(f"All motif recognition results saved to: '{MOTIF_OUTPUT_DIR}'")