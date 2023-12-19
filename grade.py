import librosa
import numpy as np
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition

def compute_mcd(mfcc1, mfcc2):
    """
    Compute the Mel Cepstral Distortion between two MFCC feature matrices.
    """
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1, mfcc2 = mfcc1[:, :min_len], mfcc2[:, :min_len]
    mcd = np.mean(np.sqrt(np.sum((mfcc1 - mfcc2) ** 2, axis=0)))
    return mcd

def compute_cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    return 1 - cosine(vec1, vec2)

def load_audio_compute_features(file_path, model):
    """
    Load audio file, compute MFCC and speaker embeddings.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    embeddings = model.encode_batch(model.preprocess([y, sr]))[0]
    return mfcc, embeddings

# Main script
def main(audio_files):
    # Load the pretrained ECAPA-TDNN model
    ecapa_tdnn = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

    mcd_scores = []
    cosine_scores = []

    for i in range(0, len(audio_files), 2):
        original_file = audio_files[i]
        tts_file = audio_files[i + 1]

        # Load audio and compute features
        mfcc_orig, emb_orig = load_audio_compute_features(original_file, ecapa_tdnn)
        mfcc_tts, emb_tts = load_audio_compute_features(tts_file, ecapa_tdnn)

        # Compute MCD
        mcd = compute_mcd(mfcc_orig, mfcc_tts)
        mcd_scores.append(mcd)

        # Compute cosine similarity
        cosine_sim = compute_cosine_similarity(emb_orig, emb_tts)
        cosine_scores.append(cosine_sim)

    # Compute averages
    average_mcd = np.mean(mcd_scores)
    average_cosine = np.mean(cosine_scores)

    # Save results to a file
    with open("results.txt", "w") as file:
        for i in range(len(mcd_scores)):
            file.write(f"Audio Pair {i+1}: MCD = {mcd_scores[i]}, Cosine Similarity = {cosine_scores[i]}\n")
        file.write(f"\nAverage MCD: {average_mcd}\nAverage Cosine Similarity: {average_cosine}\n")

    print("Results saved to results.txt")

# Example usage
# audio_files = ['audio1_orig.wav', 'audio1_tts.wav', 'audio2_orig.wav', 'audio2_tts.wav', ...]
# main(audio_files)
