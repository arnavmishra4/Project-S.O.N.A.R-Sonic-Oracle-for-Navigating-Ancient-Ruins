# Cell 3: Anomaly Detection with Isolation Forest

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM # Alternative model
import numpy as np
import os
import glob
import json # To load metadata and align anomalies

# --- RE-DEFINING GLOBAL TRANSECT LISTS FOR CELL SCOPE ---
# These must be defined here as Cell 3 runs independently and needs these variables.
ARCHAEOLOGICAL_TRANSECTS = ['BR_AC_10', 'BR_RO_05', 'BR_PA_02', 'BR_AC_07', 'BR_AC_09']
JUNGLE_TRANSECTS = ['BR_AM_04', 'BR_PA_04', 'BR_RO_03', 'BR_MT_01']
CITY_TRANSECTS = ['BR_AM_03']

# Mapping based on previous cell's implicit usage
GAN_TRANSECTS = JUNGLE_TRANSECTS
TRANSFORMER_TRANSECTS = ARCHAEOLOGICAL_TRANSECTS

# --- Configuration for Anomaly Detection Module ---
EMBEDDING_INPUT_DIR = "audio_embeddings" # Directory where VGGish embeddings are saved
ANOMALY_OUTPUT_DIR = "anomaly_results" # Directory to save anomaly scores and flags

os.makedirs(ANOMALY_OUTPUT_DIR, exist_ok=True)

# Define which transects are 'normal' for training the anomaly detection model
# IMPORTANT: Updated to only include transects for which embeddings were successfully generated in Cell 2.
# Ideally, these would be purely 'normal' jungle sites. For now, we use available embeddings.
NORMAL_TRANSECTS_FOR_TRAINING = ['BR_AC_10', 'BR_AC_07'] # Using two archaeological sites as 'normal' baseline for now

# Define which transects to apply anomaly detection to (all successfully processed ones)
TRANSECTS_TO_ANALYZE = ['BR_PA_02', 'BR_RO_05', 'BR_AC_10', 'BR_AC_07'] # Only the ones where embeddings were created

# Anomaly Detection Model Parameters
ISOLATION_FOREST_RANDOM_STATE = 42 # For reproducibility
ISOLATION_FOREST_CONTAMINATION = 0.01 # Expected proportion of anomalies in the training data (e.g., 1%)
                                         # Adjust this based on your assumption of anomaly density in baseline areas.
                                         # For OneClassSVM, nu is equivalent to contamination.

print("Cell 3: Anomaly Detection Setup Complete.")

# --- 1. Load Embeddings and Prepare Training Data ---
print("\n--- Preparing training data for Anomaly Detection Model ---")
all_normal_embeddings = []

for transect_id in NORMAL_TRANSECTS_FOR_TRAINING:
    embedding_filepath = os.path.join(EMBEDDING_INPUT_DIR, f"{transect_id}_embeddings.npy")
    if os.path.exists(embedding_filepath):
        print(f"  Loading normal embeddings for training: {transect_id}")
        embeddings = np.load(embedding_filepath)
        if embeddings.size > 0:
            all_normal_embeddings.append(embeddings)
        else:
            print(f"    Warning: No embeddings found for {transect_id}. Skipping for training.")
    else:
        print(f"  Warning: Embedding file not found for normal transect '{transect_id}'. Skipping for training.")

if not all_normal_embeddings:
    print("ERROR: No normal transect embeddings available for training. Cannot proceed with anomaly detection.")
    # Raise an error to halt execution if training data is genuinely missing
    raise ValueError("No normal transect embeddings for training. Check paths/data for NORMAL_TRANSECTS_FOR_TRAINING.")

# Concatenate all normal embeddings for training
X_train_normal = np.vstack(all_normal_embeddings)
print(f"Total number of normal embedding samples for training: {X_train_normal.shape[0]}")
print(f"Embedding dimensionality: {X_train_normal.shape[1]}")

# --- 2. Train Anomaly Detection Model (Isolation Forest) ---
print("\n--- Training Isolation Forest Model ---")
# You can choose between IsolationForest or OneClassSVM
# model = OneClassSVM(nu=ISOLATION_FOREST_CONTAMINATION, kernel="rbf", gamma="auto") # One-Class SVM
model = IsolationForest(contamination=ISOLATION_FOREST_CONTAMINATION, random_state=ISOLATION_FOREST_RANDOM_STATE)

# Fit the model to the normal data
model.fit(X_train_normal)
print("Isolation Forest model trained successfully.")

# --- 3. Apply Anomaly Detection to All Transects ---
print("\n--- Applying Anomaly Detection to all selected transects ---")

all_transect_anomaly_results = {}

for transect_id in TRANSECTS_TO_ANALYZE:
    print(f"\nProcessing transect: {transect_id}")
    embedding_filepath = os.path.join(EMBEDDING_INPUT_DIR, f"{transect_id}_embeddings.npy")
    metadata_filepath = os.path.join(SONIFIED_AUDIO_BASE_DIR, transect_id, f"{transect_id}_geospatial_metadata.json")

    if not os.path.exists(embedding_filepath):
        print(f"  Embeddings not found for {transect_id}. Skipping anomaly detection for this transect.")
        continue
    
    if not os.path.exists(metadata_filepath):
        print(f"  Metadata not found for {transect_id}. Cannot link anomalies to geospatial cells. Skipping.")
        continue

    embeddings_to_predict = np.load(embedding_filepath)
    if embeddings_to_predict.size == 0:
        print(f"  No embeddings found in file for {transect_id}. Skipping.")
        continue

    # Predict anomaly scores (lower score indicates higher anomaly)
    # Isolation Forest returns decision_function scores (negative for anomalies)
    anomaly_scores = model.decision_function(embeddings_to_predict)

    # Predict if a sample is an outlier (-1) or an inlier (1)
    # For IsolationForest, -1 is outlier, 1 is inlier.
    anomaly_predictions = model.predict(embeddings_to_predict)

    # Convert predictions to a more intuitive boolean flag: True for anomaly, False for normal
    anomaly_flags = (anomaly_predictions == -1)

    print(f"  Calculated {len(anomaly_scores)} anomaly scores for {transect_id}.")
    print(f"  Detected {np.sum(anomaly_flags)} anomalous segments ({np.sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)")

    # Load geospatial metadata to align anomalies with cells
    with open(metadata_filepath, 'r') as f:
        cell_geometries = json.load(f)

    # --- Aligning Anomaly Results with Geospatial Cells ---
    # This is a critical and potentially complex part.
    # VGGish embeddings are generated for overlapping 0.96s segments (with 0.5s hop).
    # Your sonified cells are DURATION_PER_GRID_CELL (6.0s) long.
    # We need to map which VGGish embeddings fall within which original geospatial cell's audio.

    # Calculate how many VGGish embeddings per original grid cell
    vggish_frame_length_s = 0.96
    vggish_hop_length_s = 0.5
    
    # DURATION_PER_GRID_CELL is a global constant from Cell 1. We assume its value here.
    # If this notebook were standalone, DURATION_PER_GRID_CELL would need to be defined here.
    # For now, it relies on it being run in a session where Cell 1 has executed.
    
    # Store results, linking them back to the original cell metadata
    transect_anomaly_data = []
    current_vggish_idx = 0

    for i, cell_geom in enumerate(cell_geometries):
        cell_start_ms = cell_geom['audio_start_ms']
        cell_end_ms = cell_geom['audio_end_ms']
        cell_duration_s = (cell_end_ms - cell_start_ms) / 1000.0

        # Calculate VGGish embeddings relevant to this cell's audio duration
        actual_num_vggish_embeddings_for_this_cell = int(np.floor((cell_duration_s - vggish_frame_length_s) / vggish_hop_length_s) + 1)
        if actual_num_vggish_embeddings_for_this_cell < 0: # Handle very short/silent cells that might not generate embeddings
            actual_num_vggish_embeddings_for_this_cell = 0

        # Ensure we don't go out of bounds of the extracted embeddings array
        end_vggish_idx = min(current_vggish_idx + actual_num_vggish_embeddings_for_this_cell, len(anomaly_scores))
        
        cell_vggish_scores = anomaly_scores[current_vggish_idx:end_vggish_idx]
        cell_vggish_flags = anomaly_flags[current_vggish_idx:end_vggish_idx]

        # Determine if *any* VGGish segment within this cell is anomalous
        is_cell_anomalous = np.any(cell_vggish_flags) if cell_vggish_flags.size > 0 else False
        
        # Calculate mean anomaly score for the cell (or min score if you want to emphasize worst anomaly)
        mean_cell_anomaly_score = np.mean(cell_vggish_scores) if cell_vggish_scores.size > 0 else 0.0

        transect_anomaly_data.append({
            "cell_id": i, # Or a unique ID if you have one for each cell
            "minx": cell_geom['minx'],
            "miny": cell_geom['miny'],
            "maxx": cell_geom['maxx'],
            "maxy": cell_geom['maxy'],
            "audio_start_ms": cell_geom['audio_start_ms'],
            "audio_end_ms": cell_geom['audio_end_ms'],
            "is_anomalous_flag": bool(is_cell_anomalous), # Convert to native bool
            "mean_anomaly_score": float(mean_cell_anomaly_score), # Convert to native float
            # Optionally, you can store all VGGish scores for the cell here if needed for finer analysis
            # "vggish_segment_scores": cell_vggish_scores.tolist()
        })
        current_vggish_idx = end_vggish_idx # Move to the next block of VGGish embeddings

    all_transect_anomaly_results[transect_id] = transect_anomaly_data

    # Save the anomaly results for the current transect
    output_json_filepath = os.path.join(ANOMALY_OUTPUT_DIR, f"{transect_id}_anomaly_results.json")
    with open(output_json_filepath, 'w') as f:
        json.dump(transect_anomaly_data, f, indent=4)
    print(f"  Anomaly results saved to: {output_json_filepath}")

print("\n--- Anomaly Detection Process Complete ---")
print(f"All anomaly results saved to: '{ANOMALY_OUTPUT_DIR}'")