# 🌀 S.O.N.A.R

### **Sonic Oracle for Navigating Ancient Ruins**

An AI-powered acoustic archaeology system that transforms Amazon rainforest geospatial data into sound, using machine listening to detect hidden archaeological sites beneath dense canopy.

**Author:** [Arnav Mishra](https://www.kaggle.com/arnavmishra6996)  
**Competition:** [OpenAI to Z Challenge](https://www.kaggle.com/competitions/openai-to-z-challenge)  
**Project Notebook:** [Whispers Beneath the Canopy](https://www.kaggle.com/code/arnavmishra6996/whispers-beneath-the-canopy)

---

## 🌍 Overview

**SONAR (Sonic Oracle for Navigating Ancient Ruins)** is a novel multi-stage AI pipeline that converts LiDAR terrain data, satellite imagery, and hydrological patterns into structured soundscapes — then uses deep audio embeddings, anomaly detection, and acoustic pattern matching to identify archaeological signatures invisible to traditional remote sensing.

**The Core Innovation:** Instead of analyzing geospatial data visually, SONAR **sonifies** it — mapping elevation to bass frequencies, vegetation density to melodic patterns, and terrain slope to rhythmic structures. Archaeological features produce distinctive acoustic anomalies that can be detected through machine listening.

---

## 🧩 Pipeline Architecture

### **Module 1: Geospatial Sonification Engine**

Transforms multi-modal geospatial data into interpretable audio landscapes.

**Input Data:**
- **LiDAR DTMs** — High-resolution digital terrain models (ground elevation)
- **Sentinel-2 Satellite Imagery** — Multi-band data for wet/dry seasons (NDVI, NDWI indices)
- **HydroSHEDS** — Flow direction, flow accumulation, conditioned elevation models

**Processing Pipeline:**
1. **Load & Mosaic** — Merge individual LiDAR DTM tiles into unified high-resolution transect maps
2. **Grid & Align** — Divide DTM into spatial grid cells, extract corresponding satellite and hydrological data
3. **Data-to-Sound Mapping** — Convert geospatial features to audio parameters:
   - Elevation → Bass pitch
   - Vegetation density (NDVI) → Melodic frequencies
   - Terrain slope → Rhythmic patterns
   - Water presence (NDWI) → Textural layers
4. **Anomaly Highlighting** — Overlay distinct acoustic markers on known archaeological coordinates
5. **Assembly** — Concatenate per-cell audio chunks into full-length stereo WAV files

**Output:**
- `{transect}_full_sonification_SOTA.wav` — Complete sonified landscape
- `{transect}_geospatial_metadata.json` — Audio-to-coordinate mapping
- `{transect}_visualization.png` — Input data visualization

<div align="center">
  <img src="./images/Screenshot%202025-10-26%20031458.png" alt="Geospatial Sonification Process" width="800"/>
  <p><em>Example: LiDAR elevation data transformed into acoustic landscape with archaeological anomaly markers</em></p>
</div>

---

### **Module 2: Sonic Feature Embedding (VGGish)**

Extracts deep audio representations from sonified landscapes using Google's pre-trained VGGish model.

**Model:** [Google VGGish](https://tfhub.dev/google/vggish/1) — AudioSet-trained audio embedding network

**Processing Pipeline:**
1. **Load Model** — Initialize pre-trained VGGish from TensorFlow Hub
2. **Chunk Audio** — Stream large audio files in memory-efficient 10s segments
3. **Preprocess** — Convert to mono, resample to 16kHz (VGGish requirement)
4. **Extract Embeddings** — Generate 128-dimensional feature vectors per 0.96s audio segment
5. **Aggregate** — Combine embeddings into temporal sequence arrays

**Output:** `audio_embeddings/{transect}_embeddings.npy` — 2D array of 128-dim embeddings

---

### **Module 3: Sonic Anomaly Detection (Isolation Forest)**

Identifies acoustically unusual regions that may indicate archaeological features.

**Training Strategy:**
- Train Isolation Forest exclusively on "normal" landscape embeddings (non-archaeological areas)
- Learn typical acoustic patterns of natural terrain
- Flag deviations as potential archaeological anomalies

**Processing Pipeline:**
1. **Train Baseline Model** — Fit Isolation Forest on embeddings from verified normal transects
2. **Score All Segments** — Compute anomaly scores for every audio segment across all transects
3. **Spatial Alignment** — Map time-based anomaly flags back to geospatial grid cells
4. **Cell Aggregation** — Flag cells as anomalous if any underlying audio segments are outliers
5. **Export Results** — Generate annotated geospatial anomaly maps

**Output:** `anomaly_results/{transect}_anomaly_results.json` — Geospatial cells with anomaly flags and scores

---

### **Module 4: Archaeological Signature Recognition (DTW)**

Classifies detected anomalies by matching them to known archaeological acoustic patterns.

**Approach:**
- Build reference library of acoustic "motifs" from confirmed archaeological sites
- Use Dynamic Time Warping (DTW) to measure similarity between unknown anomalies and known signatures
- Classify anomalies based on best-matching motif patterns

**Processing Pipeline:**
1. **Build Motif Library** — Extract embedding sequences from manually-defined time segments of known archaeological features
2. **Isolate Anomalous Segments** — Focus analysis on cells flagged by Module 3
3. **DTW Comparison** — Measure acoustic similarity between each anomaly and all reference motifs
4. **Pattern Matching** — Identify best-matching motif with lowest DTW distance
5. **Classification** — Assign archaeological feature type if similarity exceeds threshold

**Output:** `motif_recognition_results/{transect}_motif_recognition_results.json` — Classified anomalies with confidence scores

---

### **Module 5: Interactive Visualization (Folium)**

Generates color-coded interactive maps displaying analysis results.

**Processing Pipeline:**
1. **Data Integration** — Load geospatial metadata, anomaly flags, and motif classifications
2. **Coordinate Transformation** — Convert to WGS84 lat/lon for web mapping
3. **Map Generation** — Create interactive Folium maps with grid overlay
4. **Color Coding:**
   - 🟦 **Blue** — Normal terrain
   - 🟧 **Orange** — Detected anomaly (unmatched)
   - 🔴 **Red** — Matched archaeological signature (high confidence)

**Output:** Interactive HTML maps rendered in notebook with clickable cells showing detailed analysis

---

## 🛰️ Data Sources

| Source | Description | Resolution |
|--------|-------------|------------|
| [NASA LiDAR 2008–2018](https://www.kaggle.com/datasets/arnavmishra6996/nasa-amazon-lidar-2008-2018) | Ground elevation (Digital Terrain Models) | 1m |
| [Sentinel-2 Satellite](https://scihub.copernicus.eu/) | Multi-spectral imagery (NDVI, NDWI) | 10-20m |
| [HydroSHEDS](https://www.hydrosheds.org/) | Hydrological flow patterns and DEM | 90m |
| [VGGish (Google)](https://tfhub.dev/google/vggish/1) | Pre-trained audio embedding model | — |

---

## 🔬 Research Impact

> **SONAR demonstrates that archaeological features create detectable acoustic signatures when terrain data is sonified. This approach enables machine audition-based site discovery in regions where traditional remote sensing fails due to dense vegetation cover.**

Key innovations:
- **Cross-modal translation** — Geospatial → Acoustic domain conversion
- **Anomaly-driven discovery** — Unsupervised detection without labeled training data
- **Interpretable signatures** — DTW-based pattern matching preserves archaeological context

---

## ⚙️ Repository Structure

```
sonar-ai/
│
├── main.py                      # End-to-end pipeline orchestration
├── config.py                    # Configuration and hyperparameters
├── data_loader.py               # Multi-source data loading utilities
│
├── models/
│   ├── sonification.py          # Module 1: Data-to-audio conversion
│   ├── vggish_embedding.py      # Module 2: Audio feature extraction
│   ├── anomaly_detection.py     # Module 3: Isolation Forest
│   ├── motif_recognition.py     # Module 4: DTW pattern matching
│   └── map_visualization.py     # Module 5: Interactive mapping
│
├── images/                      # Visualization screenshots
│   ├── Screenshot 2025-10-26 031458.png
│   └── Screenshot 2025-10-26 031519.png
│
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
git clone https://github.com/arnavmishra6996/sonar-ai.git
cd sonar-ai
pip install -r requirements.txt
```

**Requirements:**
```
tensorflow>=2.13.0
tensorflow-hub
numpy
scipy
librosa
soundfile
folium
scikit-learn
rasterio
```

---

## 🚀 Usage

```bash
python main.py
```

Pipeline executes sequentially:
1. Sonification → 2. Embedding → 3. Anomaly Detection → 4. Motif Recognition → 5. Visualization

All outputs saved in `data/` with corresponding module subdirectories.

---

## 🧠 Related Kaggle Resources

| Resource | Type | Link |
|----------|------|------|
| 🌀 Whispers Beneath the Canopy | Notebook | [View](https://www.kaggle.com/code/arnavmishra6996/whispers-beneath-the-canopy) |
| 🌲 NASA Amazon LiDAR 2008–2018 | Dataset | [View](https://www.kaggle.com/datasets/arnavmishra6996/nasa-amazon-lidar-2008-2018) |
| 💧 HydroSHEDS South America | Dataset | [View](https://www.kaggle.com/datasets/arnavmishra6996/south-america-hydroshed-dataset) |
| 🎵 Sonification + VGGish Model | Model | [View](https://www.kaggle.com/models/arnavmishra6996/new-sonification-50km-vggish-anomaly-detect) |

---

## 📊 Results

<div align="center">
  <img src="./images/Screenshot%202025-10-26%20031519.png" alt="Archaeological Site Detection Results" width="800"/>
  <p><em>Interactive map showing detected archaeological anomalies (orange) and classified signatures (red) overlaid on Amazon rainforest terrain. High-confidence detections correspond to known geoglyph and earthwork locations.</em></p>
</div>

---

## 🧾 Citation

If you use SONAR in your research:

```bibtex
@misc{mishra2025sonar,
  author = {Mishra, Arnav},
  title = {S.O.N.A.R: Sonic Oracle for Navigating Ancient Ruins},
  year = {2025},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/code/arnavmishra6996/whispers-beneath-the-canopy}}
}
```

---

## 🏁 Author

**Arnav Mishra**  
AI Researcher · Machine Learning & Computational Archaeology  
Bhopal, India

[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/arnavmishra6996)

---

## 🪶 License

MIT License © 2025 Arnav Mishra
