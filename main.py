from config import *
from utils.logger import log
from models import sonification, vggish_embedding, anomaly_detection, motif_recognition, map_visualization

def main():
    log("Starting SONAR: Whispers Beneath the Canopy")

    # Stage 1: Sonification
    log("Running sonification...")
    sonification.run()

    # Stage 2: Embedding extraction
    log("Extracting VGGish embeddings...")
    vggish_embedding.run()

    # Stage 3: Anomaly detection
    log("Running anomaly detection...")
    anomaly_detection.run()

    # Stage 4: Motif recognition
    log("Performing DTW motif matching...")
    motif_recognition.run()

    # Stage 5: Visualization
    log("Generating interactive Folium maps...")
    map_visualization.run()

    log("Pipeline complete. All results saved.")

if __name__ == "__main__":
    main()
