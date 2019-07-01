

class config:
    batch_size = 8
    epochs = 100
    steps_per_epoch = 4
    validation_steps = 4 # iteration is from zero there is actually 5 steps each val/train epoch
    log_dir = r"log_dir"
    val_freq = 2