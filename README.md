# Project S.O.N.A.R

**Sonic Oracle for Navigating Ancient Ruins**

---

## Module 1: Geospatial Sonification Engine

**Dataset:**

* LiDAR DTMs: Digital Terrain Models representing ground elevation.
* Satellite Imagery: Multi-band Sentinel-2 data for wet and dry seasons, used to calculate vegetation (NDVI) and water (NDWI) indices.
* Hydrological Maps: Pre-processed HydroSHEDS data including flow direction, flow accumulation, and conditioned elevation models.

**What it does:**

* Load & Mosaic: Reads individual LiDAR DTM tiles and merges them into a single high-resolution map for the transect.
* Grid & Align: Divides the primary DTM map into a grid. For each grid cell, it aligns and extracts corresponding data from the lower-resolution satellite and hydrological layers.
* Data-to-Sound Mapping: Maps each cell's data to specific audio parameters (e.g., Elevation -> Bass Pitch, Vegetation -> Melody, Slope -> Rhythm).
* Anomaly Highlighting: Overlays a distinct, piercing sound on pre-defined coordinates corresponding to potential archaeological sites.
* Assemble & Export: Generates an audio "chunk" for each grid cell, then concatenates all chunks into a final, full-length stereo WAV file.

**Output:**

* A final, full-length .wav audio file for each transect.
* A .json metadata file linking audio timestamps to geographic coordinates.
* A .png visualization image of the input data for each transect.

---

## Module 2: Sonic Feature Embedding Module (VGGish)

**Dataset:**

* Primary Input: The final sonified .wav files generated for each transect from Module 1.
* Pre-trained Model: The Google VGGish model, loaded directly from TensorFlow Hub.

**What it does:**

* Load Model: Initializes and loads the pre-trained VGGish model.
* Chunk & Preprocess Audio: Reads the long audio files in memory-efficient chunks, converts them to mono, and resamples them to the required 16kHz.
* Extract Embeddings: Passes each preprocessed audio chunk through the VGGish model to generate a sequence of 128-dimensional feature vectors.
* Aggregate & Save: Combines the embeddings from all chunks into a single large array and saves it as a .npy file.

**Output:**

* A set of .npy files, each containing a 2D NumPy array of 128-dimensional VGGish embeddings for a corresponding transect.

---

## Module 3: Sonic Anomaly Detection Module (Isolation Forest)

**Dataset:**

* Primary Input: The VGGish embedding .npy files from Module 2.
* Training Baseline: A specific subset of embeddings designated as "normal" (e.g., from non-archaeological areas).
* Geospatial Metadata: The .json files from Module 1, used to link anomalies back to their geographic locations.

**What it does:**

* Train Model: Trains an IsolationForest model exclusively on the "normal" embeddings to learn the patterns of a typical soundscape.
* Predict on All Transects: Uses the trained model to assign an anomaly score and a binary flag (-1 for anomaly, 1 for normal) to every embedding segment.
* Align & Aggregate: Maps the time-based anomaly flags back to the space-based geospatial grid cells. A cell is flagged as anomalous if any of its underlying sonic segments are outliers.
* Save Results: Saves a new JSON file annotating each geospatial cell with an is\_anomalous\_flag and a mean anomaly score.

**Output:**

* A set of \_anomaly\_results.json files that create a map of sonically unusual locations by flagging the specific geospatial cells that correspond to anomalous sounds.

---

## Module 4: Archaeological Signature Recognition Module (DTW)

**Dataset:**

* VGGish Embeddings & Anomaly Flags: The outputs from Modules 2 and 3.
* Motif Library: A custom library built by extracting embedding sequences from manually-defined time segments of sonifications known to contain specific archaeological features.

**What it does:**

* Build Motif Library: Extracts sequences of VGGish embeddings from pre-defined time segments in known archaeological transects to create a reference library of sonic "motifs".
* Isolate Anomalous Segments: Focuses only on the geospatial cells that were previously flagged as anomalous.
* Compare Signatures with DTW: Uses Dynamic Time Warping (DTW) to measure the similarity between an unknown anomaly's signature and every known motif in the library.
* Identify Best Match & Classify: For each anomaly, it finds the known motif with the lowest DTW distance. If the distance is below a set threshold, the anomaly is "matched" and classified.
* Save Classified Results: Saves a final JSON file that annotates each anomalous cell with its best-matching motif type and a similarity score.

**Output:**

* A set of \_motif\_recognition\_results.json files that label each anomalous location with a potential archaeological signature type (e.g., "Geoglyph\_Cluster") and a similarity score.

---

## Module 5: Final Integration & Interactive Visualization Module

**Dataset:**

* Geospatial Metadata: The .json files containing the original grid cell coordinates (from Module 1).
* Motif Recognition Results: The final .json analysis files containing anomaly flags and motif classifications (from Module 4).
* AI-Generated Context: External .txt files containing pre-generated textual summaries or historical context for each transect.

**What it does:**

* Load All Processed Data: For each transect, it loads all associated output files.
* Coordinate Transformation: Converts the geographic coordinates of each grid cell into the standard Latitude/Longitude format required for web maps.
* Generate Interactive Map: Initializes a folium interactive map for each transect.
* Draw & Color-Code Cells: Draws a rectangle on the map for every grid cell, with the color determined by its final status:

  * Blue: Normal.
  * Orange: Anomalous (unmatched).
  * Red: Matched Anomaly (high-confidence).
* Display Final Results: Renders the final interactive map and its accompanying text directly into the notebook output for immediate review.

**Output:**

* A series of interactive, color-coded maps rendered in the notebook, providing a comprehensive visual summary of the entire analysis and highlighting potential archaeological sites.
