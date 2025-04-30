# evaluation/baseline_training/config.py

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_LABELS = 2
SEED = 42

# Dataset path (50k sample)
DATASET_PATH = 'datasets/training.1600000.processed.noemoticon.csv'

# Save directory
MODEL_SAVE_DIR = 'evaluation/baseline_training/trained_models'
LOG_CSV = 'evaluation/baseline_training/training_logs.csv'
