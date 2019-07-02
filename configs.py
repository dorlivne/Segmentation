

class config:
    batch_size = 4
    epochs = 1000
    steps_per_epoch = 5
    validation_steps = 4  # iteration is from zero there is actually 5 steps each val/train epoch
    log_dir = r"log_dir"
    model_path = r"saved_models/Unet"
    val_freq = 5
    stop = False

    @staticmethod
    def learning_rate_scheduler(epoch: int):
        if epoch <= 200:
            return 1e-3
        elif 200 < epoch <= 400:
            return 1e-4
        elif 400 < epoch <= 600:
            return 5e-5
        elif 600 < epoch <= 800:
            return 1e-5
        else:
            return 1e-6
