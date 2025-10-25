# Cell 5: Interactive Folium Maps within Notebook

import folium
import json
import os
import numpy as np
from pyproj import CRS, Transformer
from IPython.display import display, HTML # For displaying maps in notebooks
from config import (
    SONIFIED_AUDIO_BASE_DIR,
    MOTIF_OUTPUT_DIR as ANOMALY_MOTIF_RESULTS_INPUT_DIR, ) 
from config import BASE_DIR
CHATGPT_OUTPUT_DIR = f"{BASE_DIR}/data/chatgpt_contextualizations"
# --- Configuration & Data Paths (from your Cell 1 setup) ---
# These variables should be available in your notebook's global scope
# IF you have run Cell 1 correctly at the beginning of your session.
# If you restart your kernel and run ONLY this cell, you'll need to copy
# the definitions of SONIFIED_AUDIO_BASE_DIR, ANOMALY_MOTIF_RESULTS_INPUT_DIR,
# CHATGPT_OUTPUT_DIR, and TRANSECT_FILE_PATHS from Cell 1 to here.

SONIFIED_AUDIO_BASE_DIR = "/kaggle/input/openai-competition-dataset/sonified_outputs"
ANOMALY_MOTIF_RESULTS_INPUT_DIR = "motif_recognition_results"
CHATGPT_OUTPUT_DIR = "chatgpt_contextualizations"

# IMPORTANT: REPLACE "EPSG:XXXX" with the actual CRS reported by your Cell 5 output
# (e.g., "EPSG:5356" if that was your detected master CRS).
SOURCE_CRS = CRS("EPSG:5356") # WGS 84 / UTM zone 20S (Example, replace with your actual CRS)
TARGET_CRS = CRS("EPSG:4326") # WGS84 Lat/Lon (Standard for Folium map)

transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)

print("Setting up notebook-native mapping...")

# --- Helper Functions (adapted from app.py, no Streamlit dependencies) ---

def load_transect_data_notebook(transect_id):
    """Loads all relevant data for a given transect for notebook display."""
    data = {
        'metadata': [],
        'audio_path': None, # Audio path exists but won't be played directly in map
        'motif_results': [],
        'chatgpt_context': "No contextualization available."
    }

    # 1. Load Geospatial Metadata
    metadata_path = os.path.join(SONIFIED_AUDIO_BASE_DIR, transect_id, f"{transect_id}_geospatial_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Error reading JSON metadata for {transect_id}: {e}")
            data['metadata'] = []
    else:
        print(f"Warning: Metadata not found for {transect_id} at {metadata_path}")

    # 2. Find Sonified Audio File (path only, for reference)
    audio_base_path = os.path.join(SONIFIED_AUDIO_BASE_DIR, transect_id, f"{transect_id}_full_sonification_SOTA")
    audio_path_archeo = f"{audio_base_path}_Archaeological.wav"
    audio_path_jungle = f"{audio_base_path}_Jungle.wav"
    if os.path.exists(audio_path_archeo):
        data['audio_path'] = audio_path_archeo
    elif os.path.exists(audio_path_jungle):
        data['audio_path'] = audio_path_jungle

    # 3. Load Anomaly/Motif Results
    motif_results_path = os.path.join(ANOMALY_MOTIF_RESULTS_INPUT_DIR, f"{transect_id}_motif_recognition_results.json")
    if os.path.exists(motif_results_path):
        try:
            with open(motif_results_path, 'r') as f:
                data['motif_results'] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Error reading JSON motif results for {transect_id}: {e}")
            data['motif_results'] = []
    else:
        print(f"Warning: Motif results not found for {transect_id} at {motif_results_path}")

    # 4. Load ChatGPT Context
    chatgpt_context_path = os.path.join(CHATGPT_OUTPUT_DIR, f"{transect_id}_chatgpt_context.txt")
    if os.path.exists(chatgpt_context_path):
        try:
            with open(chatgpt_context_path, 'r', encoding='utf-8') as f:
                data['chatgpt_context'] = f.read()
        except Exception as e:
            print(f"Warning: Error reading ChatGPT context for {transect_id}: {e}")
            data['chatgpt_context'] = "Error loading contextualization."
    else:
        print(f"Warning: ChatGPT context not found for {transect_id} at {chatgpt_context_path}")

    return data

def get_motif_info_for_cell_notebook(cell_idx, motif_results):
    """Retrieves motif matching info for a specific cell_id (index)."""
    for result in motif_results:
        if result.get('cell_id') == cell_idx:
            return result
    return None

def get_transect_list_notebook():
    """Dynamically gets the list of available transects from the audio output directory."""
    if not os.path.exists(SONIFIED_AUDIO_BASE_DIR):
        print(f"Warning: Base directory for sonified audio not found: {SONIFIED_AUDIO_BASE_DIR}")
        return []
    transects = [d for d in os.listdir(SONIFIED_AUDIO_BASE_DIR) if os.path.isdir(os.path.join(SONIFIED_AUDIO_BASE_DIR, d))]
    transects.sort()
    return transects


# --- Main Loop to Generate and Display Maps ---

processed_transect_ids = get_transect_list_notebook()

if not processed_transect_ids:
    print("No transect data found to display. Please ensure all previous pipeline steps have run successfully.")
else:
    print(f"Found {len(processed_transect_ids)} transects for visualization.")

    for transect_id in processed_transect_ids:
        print(f"\n--- Displaying Visualization for Transect: {transect_id} ---")
        transect_data = load_transect_data_notebook(transect_id)

        # Initialize Folium Map
        if transect_data['metadata']:
            first_cell = transect_data['metadata'][0]
            try:
                center_x = (first_cell['minx'] + first_cell['maxx']) / 2
                center_y = (first_cell['miny'] + first_cell['maxy']) / 2
                center_lon, center_lat = transformer.transform(center_x, center_y)
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            except Exception as e:
                print(f"Warning: Error transforming coordinates for map centering for {transect_id}: {e}. Using default center.")
                m = folium.Map(location=[0, 0], zoom_start=2)
        else:
            print(f"Warning: No geospatial metadata available for {transect_id} to center map. Displaying world view.")
            m = folium.Map(location=[0, 0], zoom_start=2)

        feature_group = folium.FeatureGroup(name="Sonified Cells").add_to(m)

        # Add polygons for each grid cell
        for i, cell in enumerate(transect_data['metadata']):
            try:
                bl_lon, bl_lat = transformer.transform(cell['minx'], cell['miny'])
                tr_lon, tr_lat = transformer.transform(cell['maxx'], cell['maxy'])
                
                bounds = [[bl_lat, bl_lon], [tr_lat, tr_lon]]

                cell_motif_info = get_motif_info_for_cell_notebook(i, transect_data['motif_results'])
                is_anomalous = cell_motif_info and cell_motif_info.get('is_anomalous_flag', False)
                is_motif_matched = cell_motif_info and cell_motif_info.get('is_motif_matched', False)
                
                matched_motif_type = cell_motif_info.get('matched_motif_type', 'N/A') if cell_motif_info else 'N/A'
                mean_anomaly_score = cell_motif_info.get('mean_anomaly_score', 'N/A') if cell_motif_info else 'N/A'
                motif_similarity_score = cell_motif_info.get('motif_similarity_score', 'N/A') if cell_motif_info else 'N/A'

                fill_color = "#3186cc" # Default: Blue (Normal)
                color = "#3186cc"
                
                popup_html = f"<b>Cell ID:</b> {i}<br>" \
                             f"<b>Anomaly:</b> {'Yes' if is_anomalous else 'No'}<br>"
                
                if is_anomalous:
                    fill_color = "orange" # Anomalous cells are orange
                    color = "orange"
                    popup_html += f"<b>Anomaly Score:</b> {mean_anomaly_score:.4f}<br>"
                    if is_motif_matched:
                        fill_color = "red" # Matched anomalies are red
                        color = "red"
                        popup_html += f"<b>Matched Motif:</b> {matched_motif_type}<br>" \
                                      f"<b>Similarity:</b> {motif_similarity_score:.4f}<br>"

                # Add a rectangle with a popup for detailed info on click
                folium.Rectangle(
                    bounds=bounds,
                    color=color,
                    weight=1,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.5,
                    popup=folium.Popup(popup_html, max_width=300), # Popup on click
                    tooltip=f"Cell {i} ({'Anomaly' if is_anomalous else 'Normal'})" # Tooltip on hover
                ).add_to(feature_group)

            except Exception as e:
                print(f"Warning: Could not add cell {i} to map for {transect_id} due to coordinate transformation or other error: {e}")

        folium.LayerControl().add_to(m)
        
        # Display the map directly in the notebook output
        print(f"### Interactive Map for {transect_id}:")
        display(m)

        # Display ChatGPT context below the map
        print(f"### AI-Generated Contextualization for {transect_id}:")
        if transect_data['chatgpt_context']:
            # CORRECTED LINE: Perform the replace operation outside the f-string
            chatgpt_display_text = transect_data['chatgpt_context'].replace('\n', '<br>')
            display(HTML(f"<div style='background-color:#f9f9f9; padding:15px; border-radius:5px; margin-top:10px;'>"
                         f"<h4>Context for {transect_id}</h4>"
                         f"<p>{chatgpt_display_text}</p></div>")) # Use the pre-formatted text here
        else:
            print("No ChatGPT contextualization available for this transect.")
        print("\n" + "="*80 + "\n") # Separator for different transects

print("\nNotebook-native visualization process complete.")