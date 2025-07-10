"""
Copyright (c) Baptiste Caramiaux, Etienne Thoret
Please cite us if you use this script :)
All rights reserved

"""

from feature_extraction import utils
from feature_extraction import plotslib
from feature_extraction import auditory
from numpy.ma import append
import matplotlib.pylab as plt
from librosa import feature

import scipy.io as sio
from scipy.fft import ifft2, ifftshift
import numpy as np
import pickle
import sys
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from profiler import profile


sys.path.append(str(Path(__file__).resolve().parent))


####

# wav_file = "neutral_337-364_0356.wav"
# wav_file = "sleepiness_281-308_0308.wav"
# wav_file = "data/segment_1.wav"

rates_vec = [
    -32,
    -22.6,
    -16,
    -11.3,
    -8,
    -5.70,
    -4,
    -2,
    -1,
    -0.5,
    -0.25,
    0.25,
    0.5,
    1,
    2,
    4,
    5.70,
    8,
    11.3,
    16,
    22.6,
    32,
]
scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]

# wav_file = "../preprocess/processed_audio/segment_68.wav"
# input_dir = "../preprocess/processed_audio/"
# input_dir = "/home/christian/Desktop/C_005/THESIS/Datasets/Predi_COVID19_Fatigue_Voice_Recording/fatigue/TypeW/Type1/"
# output_dir = "extracted_features/Predi_extracted_features/"

# input_dir = "/home/christian/Desktop/C_005/THESIS/training/16Khz-training/preprocess/preprocessed_audio/preprocess_audio_erik/"
# output_dir = "extracted_features/erik_extracted_features/"

# wav_file = "Predi-COVID_0098_20200624100830_1_m4a_W_0.wav"
# wav_file = "Predi-COVID_0090_20200627194317_1_m4a_M_0.wav"
# wav_file = "Predi-COVID_0221_20200715141551_1_m4a_W_0.wav"
# audio, fs = utils.audio_data('soundTest.aiff')


def extract_features(audio_segment, fs):
    strf, auditory_spectrogram_, mod_scale, scale_rate = auditory.strf(
        audio_segment, audio_fs=fs, duration=15, rates=rates_vec, scales=scales_vec
    )

    # prints entire array
    # np.set_printoptions(threshold=np.inf)

    # initial STRF (3750, 128, 8, 22)
    # Computation of STRF (time, frequency, scale, rate) by aggregating the time dimension (axis=0) using mean

    # Compute the magnitude of the STRF and average over time
    magnitude_strf = np.abs(strf)

    # STRF (128, 8, 22)
    real_valued_strf = np.mean(magnitude_strf, axis=0)

    # print(real_valued_strf)  ## print entire array of STRF
    return real_valued_strf, fs


# feature extraction for segmented audio in specific directory
def feature_extract_dir(input_dir: Path, output_dir: Path):
    for filename in input_dir.iterdir():
        print(f"Processing file: {filename}")

        audio_file = input_dir / filename

        audio_file, fs = utils.audio_data(audio_file)
        real_valued_strf, fs = extract_features(audio_file, fs)

        output_file = output_dir / f"{filename.stem}_strf.pkl"
        strf_data = {
            "strf": real_valued_strf,
            "fs": fs,
        }

        with open(output_file, "wb") as f:
            pickle.dump(strf_data, f)

        print(f"Saved output to: {output_file}")


def process_segment(i, segment, output_dir_str, sample_rate):
    print(f"Processing Segment {i + 1}")

    real_valued_strf, fs = extract_features(segment, sample_rate)

    return real_valued_strf


@profile
def feature_extract_segments(segment_audio_arr, output_dir: Path, sample_rate):

    with ProcessPoolExecutor(max_workers=6) as executor:
        # Submit in order and keep the futures in the same order
        futures = [
            executor.submit(process_segment, i, segment, str(output_dir), sample_rate)
            for i, segment in enumerate(segment_audio_arr)
        ]

        # Retrieve results in the same order as submitted
        features = [future.result() for future in futures]

    return features


# feature_extract_segments(input_dir, output_dir, sample_rate)

# with open("strf_data_new.pkl", "rb") as f:  # 'rb' mode for reading in binary
#     loaded_strf_data = pickle.load(f)

# print(strf)

#     scale_rate,
#     aspect="auto",
#     origin="lower",
#     extent=[rates_vec[0], rates_vec[-1], scales_vec[0], scales_vec[-1]],
# )
# plt.colorbar(label="Modulation Energy")
# plt.xlabel("Temporal Modulation Rate (Hz)") plt.ylabel("Spectral Modulation Scale (cyc/oct)")
# plt.title("Rate-Scale Representation of Spectro-Temporal Modulation")
# plt.show()


# plt.imshow(
#     np.transpose(auditory_spectrogram_[:][1:80]),
#     aspect="auto",
#     interpolation="gaussian",
#     origin="lower",
# )
# plt.show()
# print(auditory_spectrogram_)
# print(strf.shape)

# avgvec = plotslib.strf2avgvec(strf)
# strf_scale_rate, strf_freq_rate, strf_freq_scale = plotslib.avgvec2strfavg(
#     avgvec, nbScales=len(scales_vec), nbRates=len(rates_vec)
# )


# plt.figure(figsize=(8, 6))
# plt.imshow(
#     strf_scale_rate,
#     aspect="auto",
#     origin="lower",
#     extent=[rates_vec[0], rates_vec[-1], scales_vec[0], scales_vec[-1]],
#     interpolation="gaussian",
#     cmap="viridis",
# )
#
# plt.colorbar(label="Modulation Energy (Amplitude)")
# plt.xlabel("Temporal Modulation Rate (Hz)")
# plt.ylabel("Spectral Modulation Scale (cyc/oct)")
# plt.title("Rate-Scale Representation (strf_scale_rate)")
# plt.show()

# scale-rate, freq-rate, freq-scale
# plotslib.plotStrfavgEqual(
#     strf_scale_rate, strf_freq_rate, strf_freq_scale, cmap="viridis"
# )

# strf_data = {
#     "strf": real_valued_strf,
#     "fs": 44100,
# }
#
# with open("strf_data_new.pkl", "wb") as f:
#     pickle.dump(strf_data, f)
#
# with open("strf_data_new.pkl", "rb") as f:  # 'rb' mode for reading in binary
#     loaded_strf_data = pickle.load(f)
#
# print(loaded_strf_data["strf"])
