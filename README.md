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


<pre>
```
language:nl
datasets: facebook/voxpopuli
pipeline_tag: text-to-speech
```
 </pre>

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# TTS_German_Speecht5_finetuned_voxpopuli_nl

This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the facebook/voxpopuli dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4593



## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.5248        | 4.3   | 1000 | 0.4792          |
| 0.5019        | 8.61  | 2000 | 0.4663          |
| 0.4937        | 12.91 | 3000 | 0.4609          |
| 0.4896        | 17.21 | 4000 | 0.4593          |


### Framework versions

- Transformers 4.30.2
- Pytorch 2.0.1+cu118
- Datasets 2.13.1
- Tokenizers 0.13.3
