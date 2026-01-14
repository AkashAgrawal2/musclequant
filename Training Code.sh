TRAIN_DIR="/Users/akashagr/Desktop/Training Data/Utrophin Background"

cellpose \
  --train \
  --use_gpu \
  --gpu_device mps \
  --dir "$TRAIN_DIR" \
  --pretrained_model cyto2 \
  --chan 0 --chan2 0 \
  --mask_filter "_masks" \
  --learning_rate 1e-5 \
  --n_epochs 100 \
  --train_batch_size 4 \
  --min_train_masks 5 \
  --save_every 25 \
  --model_name_out "Utrophin_p14_muscle" \
  --verbose
