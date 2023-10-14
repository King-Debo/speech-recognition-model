# Import the necessary modules
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import srt

# Define a function to calculate the word error rate between two transcripts
def calculate_word_error_rate(original_transcript, predicted_transcript):
  # Convert the transcripts to lowercase and split them into words
  original_words = original_transcript.lower().split()
  predicted_words = predicted_transcript.lower().split()
  # Use the Levenshtein distance algorithm to find the minimum number of edits (insertions, deletions, or substitutions) to transform one transcript into another
  distance_matrix = np.zeros((len(original_words) + 1, len(predicted_words) + 1))
  for i in range(len(original_words) + 1):
    distance_matrix[i, 0] = i
  for j in range(len(predicted_words) + 1):
    distance_matrix[0, j] = j
  for i in range(1, len(original_words) + 1):
    for j in range(1, len(predicted_words) + 1):
      if original_words[i - 1] == predicted_words[j - 1]:
        distance_matrix[i, j] = distance_matrix[i - 1, j - 1]
      else:
        distance_matrix[i, j] = min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]) + 1
  # Calculate the word error rate as the ratio of the minimum number of edits to the length of the original transcript
  word_error_rate = distance_matrix[-1, -1] / len(original_words)
  return word_error_rate

# Define a function to generate captions and subtitles for an audio recording using a speech recognition model
def generate_captions_and_subtitles(audio_path, model, language="en", frame_rate=25):
  # Load the audio file and convert it to a numpy array
  audio_data, sample_rate = librosa.load(audio_path)
  audio_data = audio_data.astype(np.float32)
  # Predict the transcript for the audio recording using the speech recognition model
  transcript = model.predict(audio_data)
  # Split the transcript into sentences using punctuation marks
  sentences = tf.keras.preprocessing.text.text_to_word_sequence(transcript, filters=".,;:!?")
  # Create a list of subtitles using the srt module
  subtitles = []
  start_time = 0 # The start time of the first subtitle in seconds
  end_time = 0 # The end time of the first subtitle in seconds
  for sentence in sentences:
    # Estimate the duration of the sentence based on the number of words and the average speaking rate
    duration = len(sentence.split()) / (150 / 60) # The average speaking rate is assumed to be 150 words per minute
    end_time = start_time + duration # The end time of the subtitle is the start time plus the duration
    # Create a subtitle object with the sentence, the start time, and the end time
    subtitle = srt.Subtitle(index=len(subtitles) + 1, content=sentence, start=srt.srt_timestamp_to_timedelta(start_time), end=srt.srt_timestamp_to_timedelta(end_time))
    subtitles.append(subtitle)
    start_time = end_time # The start time of the next subtitle is the end time of the current subtitle
  # Convert the list of subtitles to a string using the srt module
  subtitles_string = srt.compose(subtitles)
  return subtitles_string

# Define a function to save and load a speech recognition model using TensorFlow SavedModel format
def save_and_load_model(model, model_path):
  # Save the model to a specified path using the tf.saved_model.save function
  tf.saved_model.save(model, model_path)
  # Load the model from the specified path using the tf.saved_model.load function
  loaded_model = tf.saved_model.load(model_path)
  return loaded_model

