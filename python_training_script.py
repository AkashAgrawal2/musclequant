from pathlib import Path

import torch
from cellpose import io, models, train

# 1) Paths
train_dir = Path("/Path/to/folder")
save_dir = train_dir / "models"
save_dir.mkdir(parents=True, exist_ok=True)

# 2) Load training data (images + _masks)
train_data, train_labels, _, _, _, _ = io.load_train_test_data(
    train_dir=str(train_dir),
    mask_filter="_masks",
    look_one_level_down=False,
)

print(f"Loaded {len(train_data)} training images")

# 3) Choose device: MPS if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    gpu = True
    print("Using MPS for training")
else:
    device = torch.device("cpu")
    gpu = False
    print("Using CPU for training")

# 4) Build model from a pretrained backbone, with safe dtype
#    - use_bfloat16=True on GPU/MPS is fine (Cellpose will auto-cast on MPS during training)
#    - use_bfloat16=False on CPU avoids your "mse_cpu not implemented for 'BFloat16'" error
use_bfloat16 = False if device.type == "cpu" else True

model = models.CellposeModel(
    gpu=gpu,
    device=device,
    pretrained_model="cpsam",  # or "cyto2" if you prefer the smaller CP3 backbone
    use_bfloat16=use_bfloat16,
)

print("Model device:", model.device, "dtype:", next(model.net.parameters()).dtype)

# 5) Train
cpmodel_path = train.train_seg(
    net=model.net,
    train_data=train_data,
    train_labels=train_labels,
    learning_rate=1e-5,
    n_epochs=100,
    batch_size=4,
    weight_decay=0.1,
    min_train_masks=5,
    save_path=str(save_dir),
    save_every=25,
    model_name="Utrophin_p14_muscle",
)

print("Saved trained model to:", cpmodel_path)



# python -m cellpose      to run the cellpose gui
