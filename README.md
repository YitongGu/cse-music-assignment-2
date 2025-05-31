# CSE Music Assignment 2

This repository contains code for music generation using deep learning techniques.

## Task 1: Symbolic Music Generation

Task 1 implements symbolic music generation using a sequence of tokens, trained on the MAESTRO v3.0.0 dataset.

### Prerequisites

- Python 3.x
- PyTorch
- pretty_midi
- muspy
- numpy
- tqdm

You can install the required packages using pip:

```bash
pip install torch pretty_midi muspy numpy tqdm
```

### Dataset

The dataset is provided in the `music_data.zip` file. Extract it to get the MAESTRO v3.0.0 dataset:

```bash
unzip music_data.zip
```

### Running the Code

To run the music generation model, use the following command:

```bash
python task1.py --midi_dir music_data/maestro-v3.0.0 --out_dir ./exp/task1 --n_epochs 10 --batch_size 8 --seq_len 1024 --generate_len 512
```

#### Command-line Arguments

- `--midi_dir`: Path to the MAESTRO dataset directory (required)
- `--out_dir`: Directory to save model checkpoints and generated samples (required)
- `--n_epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 1e-3)
- `--seq_len`: Sequence length for training (default: 512)
- `--generate_len`: Length of generated sequences (default: 1024)
- `--device`: Device to use for training ('cuda' or 'cpu', default: 'cuda' if available)
- `--clear_cache`: Flag to clear the encoded cache (optional)

### Output

The script will:
1. Train the model and save checkpoints in the specified output directory
2. Generate a sample MIDI file (`sample.mid`) using the best model checkpoint
3. Save the model checkpoints as `best.pt` and `last.pt`

### Model Architecture

The model uses an LSTM-based architecture with:
- Embedding layer
- 3-layer LSTM
- Linear output layer
- Dropout for regularization

The model processes music as a sequence of tokens representing:
- Time shifts (T tokens)
- Pitch values (P tokens)
- Duration values (D tokens) 