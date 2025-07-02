# Inherit from the DINO 4-scale Swin-L model configuration
_base_ = ['DINO_4scale_swin.py']

# === Dataset-specific modifications ===

# Set the number of classes in your dataset. This remains correct.
# Your classes are 'elektrik-direği-bbox' and 'trafo-bbox'.
num_classes = 2

# The DeNoising (DN) component needs to know the size of the label set.
# When training from scratch, this MUST match your num_classes.
dn_labelbook_size = 2

# === Training parameter modifications for "From Scratch" ===

# Adjust batch size based on your GPU memory. Swin-L is larger than ResNet-50.
batch_size = 2

# Training from scratch requires significantly more epochs than fine-tuning.
# 150 is a reasonable starting point. You can adjust this later.
epochs = 1000

# Adjust the learning rate schedule to match the new epoch count.
# A common practice is to drop the LR at ~80% of the training duration.
lr_drop = 700

# Define an interval for saving checkpoints.
save_checkpoint_interval = 100