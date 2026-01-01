# Generative AI Assignment 2: Deep Learning Models for Translation & Image Generation

This repository contains implementations for several advanced deep learning models, as part of the FALL25 Generative AI Assignment 2. The project covers both Natural Language Processing (NLP) and Computer Vision (CV) tasks, including translation, image generation, and diffusion models.

## Contents

- **Transformer for English to Urdu Translation**  
  Implements a transformer-based sequence-to-sequence model for translating English sentences to Urdu. Includes dataset downloading, preprocessing, model architecture, training, and evaluation.

- **BERT Fine-Tuning for Translation**  
  Fine-tunes a BERT model for parallel corpus translation tasks (English ↔ Urdu). Covers data loading, preprocessing, and model adaptation for translation.

- **CycleGAN for Image-to-Image Translation**  
  Implements CycleGAN for unpaired image-to-image translation. Includes model definitions (Generator, Discriminator, Residual Blocks), training loops, and sample results.

- **Image Generation with Diffusion Models**  
  Explores image generation using Denoising Diffusion Probabilistic Models (DDPM). Includes dataset preparation (CIFAR-10), noise scheduling, forward/reverse diffusion processes, and visualization.

- **SitReg Diffusion Model**  
  Advanced implementation of a diffusion model for image generation, with custom architecture, training routines, and evaluation on CIFAR-10.

## Structure

- `transformer_for_english_to_urdu.ipynb`  
  Transformer-based translation model.

- `bert_fine_tuning_for_translation.ipynb`  
  BERT fine-tuning for translation.

- `cycle_Gans_(1)_(1).ipynb`  
  CycleGAN for image translation.

- `Q3_image_generation_with_diffusion_models.ipynb`  
  Diffusion models for image generation.

- `sit_reg_diffusion_model.ipynb`  
  Custom diffusion model implementation.

## Getting Started

1. **Clone the repository**  
   ```
   git clone <repo-url>
   ```

2. **Install dependencies**  
   - Python 3.8+
   - PyTorch, torchvision
   - KaggleHub (for dataset download)
   - Other packages as required in each notebook

3. **Run Notebooks**  
   Open each notebook in Jupyter or VS Code and run the cells sequentially.

## Assignment Reference

See the attached PDF (`FALL25_GenAI_Assignment_2.pdf`) for detailed assignment instructions and requirements.

## License

This project is for educational purposes.
