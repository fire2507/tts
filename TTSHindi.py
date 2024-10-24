from TTS.tts.datasets import load_tts_dataset
from TTS.tts.models.tacotron2 import Tacotron2

# Load the regional language dataset
dataset_path = r'C:\Users\user\Downloads\cv-corpus-19.0-2024-09-13-hi.tar.gz\cv-corpus-19.0-2024-09-13\hi\clips'
train_data, val_data = load_tts_dataset(dataset_path, ap)

# Load pre-trained multi-language model
model = Tacotron2.from_pretrained("tts_models/multi/language/tacotron2")

# Fine-tune the model
model.train(train_data, val_data, config=training_config)

# Save the model
model.save(r'C:\Users\user\Desktop\Practice\IITR')

speech = model.synthesize("Sample sentence in regional language.")
ap.save_wav(speech, "regional_example.wav")
