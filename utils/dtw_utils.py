from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def dtw_distance(seq1, seq2):
    """Compute DTW distance between two embedding sequences."""
    if len(seq1) == 0 or len(seq2) == 0:
        return np.inf
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

def motif_match(candidate_embeddings, motif_embeddings_list, threshold=75):
    """Find best matching motif type and distance."""
    best_match = {"motif_type": "No_Match", "distance": np.inf}
    for motif_type, motif_list in motif_embeddings_list.items():
        for motif in motif_list:
            dist = dtw_distance(candidate_embeddings, motif)
            if dist < best_match["distance"]:
                best_match = {"motif_type": motif_type, "distance": dist}
    best_match["is_match"] = best_match["distance"] < threshold
    return best_match
