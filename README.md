# Text-to-Speech TTS Model Training for German Language

Welcome to my repository where I share my German Text-to-Speech (TTS) model trained using Hugging Face Transformers.

You can find the trained model on my Hugging Face Model Hub: [Salama1429/TTS_German_Speecht5_finetuned_voxpopuli_nl](https://huggingface.co/Salama1429/TTS_German_Speecht5_finetuned_voxpopuli_nl)

## Model Usage

To use the TTS model, you can follow these steps:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-to-speech", model="Salama1429/TTS_German_Speecht5_finetuned_voxpopuli_nl")

# Input your desired text
output_audio = pipe(desired_text)
```

```yaml
language: nl
datasets: facebook/voxpopuli
pipeline_tag: text-to-speech
```


## About the Model

The `TTS_German_Speecht5_finetuned_voxpopuli_nl` model is a fine-tuned version of the [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) model on the `facebook/voxpopuli` dataset. During evaluation, the model achieved the following results:

- Loss: 0.4593

## Training Procedure

### Training Hyperparameters

The model was trained using the following hyperparameters:
- Learning Rate: 1e-05
- Training Batch Size: 4
- Evaluation Batch Size: 2
- Seed: 42
- Gradient Accumulation Steps: 8
- Total Train Batch Size: 32
- Optimizer: Adam with betas=(0.9, 0.999) and epsilon=1e-08
- Learning Rate Scheduler Type: Linear
- Learning Rate Scheduler Warmup Steps: 500
- Training Steps: 4000

### Training Results

| Training Loss | Epoch | Step  | Validation Loss |
|:-------------:|:-----:|:-----:|:---------------:|
| 0.5248        | 4.3   | 1000  | 0.4792          |
| 0.5019        | 8.61  | 2000  | 0.4663          |
| 0.4937        | 12.91 | 3000  | 0.4609          |
| 0.4896        | 17.21 | 4000  | 0.4593          |

### Framework Versions

- Transformers 4.30.2
- PyTorch 2.0.1+cu118
- Datasets 2.13.1
- Tokenizers 0.13.3

