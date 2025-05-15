import io
import zipfile
from feature_extraction.strf_analyzer import STRFAnalyzer
from feature_extraction.run_extraction import feature_extract_segments
from preprocess.preprocess import preprocess_audio
import sys
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pydub import AudioSegment
from pathlib import Path
from http import HTTPStatus
from enum import Enum
from dataclasses import dataclass
import pickle
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

sys.path.append("preprocess/")
sys.path.append("feature_extraction/")

app = Flask(__name__)
CORS(app)
uploads_path = Path("tmp/uploads")

strf_analyzer = STRFAnalyzer()


@app.route("/")
def home():
    return "Flask server is running."


class SD_Class(Enum):
    NSD = 0
    SD = 1


@dataclass
class Classification:
    sd: SD_Class
    classes: list[SD_Class]
    scores: list[float]
    confidence_score: float
    result: str
    is_success: bool
    # other fields here

    def into_json(self):
        return jsonify(
            {
                "class": self.sd.value,
                "classes": [c.value for c in self.classes],
                "scores": self.scores,
                "confidence_score": self.confidence_score,
                "result": self.result,
            }
        )


@app.route('/plots/<path:filename>')
def get_plot(filename):
    print(f"Requesting plot: {filename}")
    path = Path('feature_analysis/strf_plots/').resolve(strict=True)
    return send_from_directory(path, filename)

@app.route('/segments')
def Segments():
    segments_dir = Path('preprocess/preprocessed_audio/processed_audio/segmented_audio')

    # Construct an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for _, file in enumerate(segments_dir.glob('segment_*.wav')):
            zip_file.write(file, arcname=file.name)

    # reset buffer pointer back to 0
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        download_name='segments.zip',
        as_attachment=True,
    )


@app.route("/upload", methods=["POST"])
def Upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), HTTPStatus.BAD_REQUEST

    audio_file = request.files["audio"]
    if audio_file.filename:
        uploads_path.mkdir(parents=True, exist_ok=True)

        file_path = uploads_path / secure_filename(audio_file.filename)
        audio_file.save(file_path)

        wav_file = convertWAV(file_path)
        clf = classify(wav_file)

        return (
            clf.into_json(),
            HTTPStatus.OK if clf.is_success else HTTPStatus.BAD_REQUEST,
        )

    return (
        jsonify({"error": "There was a problem saving the file"}),
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def predict_features(features, svm, pca):
    # error when audio is less than 15 secs
    if not features:
        print("!!!!!!!!!!error: no features accepted!!!!!!!!!!")
        print("Make sure the audio recording length is atleast 15 seconds.")
        is_success = False
        return 0, 0, 0.0, is_success
    pre_counter = 0
    post_counter = 0
    avg_confidence_score = 0.0
    classes = []
    confidence_scores = []

    for i, feature in enumerate(features):
        print(f"Processing feature {i + 1}")

        # Flatten and normalize the feature array
        feature_flattened = np.asarray(feature).flatten()
        feature_normalized = (
            feature_flattened / np.max(np.abs(feature_flattened))
            if np.max(np.abs(feature_flattened)) != 0
            else feature_flattened
        )

        # Reshape the feature array for PCA transformation
        feature_reshaped = feature_normalized.reshape(1, -1)

        expected_features = pca.components_.shape[1]
        if feature_flattened.shape[0] != expected_features:
            raise ValueError(
                f"Feature mismatch! Expected {expected_features} features, but got {
                    feature_flattened.shape[0]
                }."
            )

        # Apply PCA transformation
        feature_pca = pca.transform(feature_reshaped)

        # predict class label
        y_pred = svm.predict(feature_pca)

        # 0 - post. 1 - pre
        # print(f"bacc: {balanced_accuracy_score(svm.classes_, y_pred)}")
        # print(f"best params of the trained model: {svm.best_params_}")
        # print("All classes in the model:", svm.classes_)
        # print("Predicted class:", y_pred)

        # print(f"post counts: {post_counter}")
        # print(f"pre counts: {pre_counter}")

        # if hasattr(svm, "decision_function"):
        #     decision_scores = svm.decision_function(feature_pca)
        #     probs = softmax(decision_scores)

        #     probs = svm.predict_proba(feature_pca)
        # else:
        #     raise AttributeError(
        #         "SVM model does not support decision_function or predict_proba."
        #     )

        # This is where the computation of Confidence Score occurs
        # logits = np.array([1.0, 0.1])
        # probs = softmax(logits)
        # confidence = np.max(probs)

        # probs = svm.predict_proba(feature_pca)
        # confidence = np.max(probs) # probability of the predicted class

        decision_scores = svm.decision_function(feature_pca)
        confidence = np.abs(decision_scores)

        assert len(confidence) == 1
        confidence = confidence[0]

        # decision_scores = svm.decision_function(feature_pca)[0]
        # min_dist, max_dist = np.min(decision_scores), np.max(decision_scores)
        #
        # if max_dist == min_dist:
        #     confidence = 0.5  # neutral confidence if all scores are identical
        # else:
        #     confidence = (decision_scores - min_dist) / (max_dist - min_dist)

        # confidence = confidence

        confidence_scores.append(confidence)

        print(f"confidence score: {confidence}")

        # Update counters based on prediction
        print(f"SVM classes: {svm.classes_}")
        if y_pred == svm.classes_[0]:
            post_counter += 1
            classes.append(SD_Class.SD)
        elif y_pred == svm.classes_[1]:
            pre_counter += 1
            classes.append(SD_Class.NSD)

        print(f"Predicted class for feature {i + 1}: {y_pred}")
        print(f"Confidence score: {confidence}")

        avg_confidence_score += confidence

    try:
        avg_confidence_score = avg_confidence_score / len(features)
    except ZeroDivisionError:
        if hasattr(svm, "decision_function"):
            decision_scores = svm.decision_function(feature_pca)
            probs = softmax(decision_scores)
        elif hasattr(svm, "predict_proba"):
            probs = svm.predict_proba(feature_pca)
        else:
            raise AttributeError(
                "SVM model does not support decision_function or predict_proba."
            )

        # avg_confidence_score = np.max(probs)

    print(f"Pre (non-sleep-deprived) features counts: {pre_counter}")
    print(f"Post (sleep-deprived) features counts: {post_counter}")
    print(f"average CFS: {avg_confidence_score}")

    is_success = True

    return pre_counter, post_counter, classes, confidence_scores, avg_confidence_score, is_success


def classify(audio_path: Path) -> Classification:
    """
    Predict the class labels for the given STM features array of 3D using the trained SVM and PCA models.

    Args:
        features (list): List of feature arrays (e.g., STRF features).
        svm_path (str): Path to the trained SVM model (.pkl file).
        pca_path (str): Path to the trained PCA model (.pkl file).
    """
    svm_path = Path(
        # "$HOME/Research/Sleep Deprivation Detection using voice/output/pop_level/svm_fold_4.pkl"
        "./svm_with_pca_fold_4.pkl"
    )

    test_sample_path = Path(
        # "~/Research/Sleep Deprivation Detection using voice/strf_data_new.pkl"
        # "~/Research/Sleep Deprivation Detection using voice/dataset/osf/stmtf/strf_session_post_subjectNb_01_daySession_01_segmentNb_0.pkl"
        # "~/github/16Khz-models/feature_extraction/gian_data_new.pkl"
        # "~/github/16Khz-models/feature_extraction/pkls/segment_72_strf.pkl"
        "./strf_data_new.pkl"
    )

    # Load the SVM and PCA models using pickle
    with open(svm_path, "rb") as f:
        data = pickle.load(f)
    svm = data["svm"]
    pca = data["pca"]
    # Define the output directory, if necessary to be stored
    output_dir_processed = Path(
        "preprocess/preprocessed_audio/processed_audio/")
    output_dir_features = Path("feature_extraction/extracted_features/feature")
    output_dir_segmented = output_dir_processed / "segmented_audio"

    # Preprocess
    segments, sr = preprocess_audio(audio_path, output_dir_processed)

    # Compute and save STRFs
    avg_scale_rate, avg_freq_rate, avg_freq_scale = strf_analyzer.compute_avg_strf(
        output_dir_segmented
    )
    strf_analyzer.save_plots(
        avg_scale_rate,
        avg_freq_rate,
        avg_freq_scale,
        Path("feature_analysis/strf_plots"),
    )

    # Print details
    print(f"Number of segments: {len(segments)}")
    print(f"Sampling rate: {sr} Hz")

    # Feature Extraction
    features = feature_extract_segments(segments, output_dir_features, sr)
    print("Feature Extraction Complete.")

    # test_sample = pickle.load(test_sample_path)
    # with open(test_sample_path, "rb") as f:
    #     test_sample = pickle.load(f)

    # print(type(test_sample), test_sample)
    # np.set_printoptions(threshold=np.inf)
    #
    # magnitude_strf = np.abs(test_sample)
    #
    # # STRF (128, 8, 22)
    # test_sample = np.mean(magnitude_strf, axis=0)
    # print(test_sample["strf"])

    pre_count, post_count, classes, confidence_scores, avg_confidence_score, is_success = predict_features(
        features, svm, pca
    )

    print(f"\nsuccess: {is_success}\n")

    if post_count > pre_count:
        return Classification(
            sd=SD_Class.SD,
            scores=confidence_scores,
            classes=classes,
            confidence_score=avg_confidence_score,
            result="You are sleep deprived.",
            is_success=is_success,
        )
    else:
        return Classification(
            sd=SD_Class.NSD,
            scores=confidence_scores,
            classes=classes,
            confidence_score=avg_confidence_score,
            result="You are not sleep deprived.",
            is_success=is_success,
        )


def convertWAV(audio: Path) -> Path:
    wav = audio.with_suffix(".wav")
    file = AudioSegment.from_file(audio)
    file.export(wav, format="wav")

    audio.unlink()
    return Path(wav)
