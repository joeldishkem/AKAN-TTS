Asante Twi Text-to-Speech (TTS)

Overview
This repository contains a working Text-to-Speech (TTS) pipeline developed for the UG Human-Computer Interaction Lab (Department of Computer Science)[cite: 1]. The project focuses on generating intelligible and natural-sounding Asante Twi speech from text inputs using deep learning architecture adapted in a cloud-based Python environment[cite: 1].

---

Repository Structure
* `Akan_TTS_Pipeline.ipynb` - The primary Google Colab runnable notebook containing data preprocessing, model loading, fine-tuning/adaptation architecture, and inference steps[cite: 1].
* `audio_samples/` - Folder containing the 5 required generated `.wav` audio files and their corresponding text prompts[cite: 1, 2].
* `Technical_Report.docx` ( or `.txt`) - The complete 4-page technical report covering data, methodology, evaluation, and linguistic limitations[cite: 2].

---

Environment Setup & Dependencies
To run this pipeline successfully, ensure you have Python installed along with the required Hugging Face and audio processing libraries. You can install them via pip:

pip install --upgrade transformers accelerate torch soundfile scipy

---

How to Run Inference
1. Open the `Akan_TTS_Pipeline.ipynb` notebook in Google Colab (or your local Jupyter environment).
2. Run the installation and import cells.
3. Use the following code snippet to generate speech from any unseen Asante Twi text:

from transformers import pipeline
import scipy.io.wavfile as wav

# Load the Akan TTS model
tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-aka")

# Input your Asante Twi text
twi_text = "Akwaaba. Wo ho te sen?"
output = tts_pipeline(twi_text)

# Save the output audio
wav.write("output.wav", output["sampling_rate"], output["audio"])

---

Acknowledgments & Attribution
* Base Architecture & Model: Meta's Massively Multilingual Speech (MMS) VITS model (`facebook/mms-tts-aka`)[cite: 1].
* Frameworks: Hugging Face `transformers`, PyTorch, and Google Colab cloud infrastructure.
* Author: Joel Makafui Mawuko (PhD Candidate, University of Ghana)
