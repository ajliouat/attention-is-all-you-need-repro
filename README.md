# Transformer Implementation in PyTorch

This repository contains a state-of-the-art implementation of the Transformer architecture from the paper "Attention Is All You Need" by Vaswani et al. (2017). The code is written in PyTorch and is designed to be run on Google Colab for optimal performance.

## Features

- Implements the Transformer architecture with multi-head attention, positional encoding, and separate encoder and decoder layers.
- Utilizes advanced techniques such as AdamW optimizer, linear learning rate scheduler with warmup, and label smoothing.
- Includes a custom dataset class for handling parallel source and target sequences.
- Generates masks for source and target sequences to handle variable-length inputs and prevent future information leakage in the decoder.
- Provides a comprehensive training loop with progress tracking and loss reporting.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Transformers library (Hugging Face)

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/ajliouat/attention-is-all-you-need-repro.git
   ```

2. Prepare your dataset:
   - Create vocabulary files for source and target languages (`src_vocab.txt` and `tgt_vocab.txt`).
   - Preprocess your parallel corpus and save the source and target sequences as `src_data.txt` and `tgt_data.txt`.

3. Set the hyperparameters:
   - Modify the hyperparameters in the code according to your requirements and the recommendations from the paper.

4. Run the code:
   - Open the Colab notebook (`transformer.py`) and upload it to your Google Drive.
   - Mount your Google Drive in the notebook to access the dataset and vocabulary files.
   - Install the required dependencies (PyTorch and Transformers) in the notebook.
   - Run the notebook cells to train the Transformer model on your dataset.

## Model Architecture

The Transformer model consists of an encoder and a decoder, each composed of multiple layers. The encoder processes the source sequence, while the decoder generates the target sequence. The key components of the architecture are:

- Positional Encoding: Adds positional information to the input embeddings.
- Multi-Head Attention: Allows the model to attend to different parts of the input sequence.
- Feed-Forward Networks: Applies non-linear transformations to the attention outputs.
- Residual Connections and Layer Normalization: Facilitates training of deep models.

## Training

The model is trained using the Adam optimizer with a linear learning rate scheduler and warmup. The training loop utilizes teacher forcing, where the decoder receives the ground truth target sequence as input during training. The loss is computed using cross-entropy with label smoothing.

## Acknowledgments

This implementation is based on the paper "Attention Is All You Need" by Vaswani et al. (2017) and draws inspiration from various open-source implementations and tutorials.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

## Contact

For questions or feedback, please contact [a.jliouat@yahoo.fr](mailto:a.jliouat@yahoo.fr).
