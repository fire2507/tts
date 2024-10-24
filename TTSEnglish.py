from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_dataset
from TTS.tts.utils.text import phoneme_to_sequence
from TTS.tts.models.tacotron2 import Tacotron2

# Setup audio processor and load the dataset
ap = AudioProcessor()

# Load dataset (replace with your dataset path)
dataset_path = r'C:\Users\user\Downloads\archive (3).zip\LJSpeech-1.1\wavs'
train_data, val_data = load_tts_dataset(dataset_path, ap)

# Modify phonemes for better pronunciation of technical jargon
def custom_phoneme_map(word):
    if word == "API":
        return phoneme_to_sequence("A P I")
    elif word == "CUDA":
        return phoneme_to_sequence("C U D A")
    else:
        return phoneme_to_sequence(word)

# Initialize the model for fine-tuning
model = Tacotron2.from_pretrained("tts_models/en/ljspeech/tacotron2-DCA")

# Set training parameters
training_config = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100
}

# Fine-tune the model
model.train(train_data, val_data, config=training_config)

# Save the fine-tuned model
model.save(r'C:\Users\user\Desktop\Practice\IITR')

# Generate a speech sample using the fine-tuned model
speech = model.synthesize("The CUDA library accelerates computing performance.")
ap.save_wav(speech, "cuda_example.wav")

# Evaluate MOS score using native speakers or automate using a pre-built MOS function
import time

start_time = time.time()
speech = model.synthesize("The REST API allows server-client communication.")
end_time = time.time()

print("Inference Time:", end_time - start_time)
