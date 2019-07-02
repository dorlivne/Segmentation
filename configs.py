

class config:
    batch_size = 4
    epochs = 400
    steps_per_epoch = 5
    validation_steps = 4  # iteration is from zero there is actually 5 steps each val/train epoch
    log_dir = r"log_dir"
    model_path = r"saved_models/Unet"
    val_freq = 5
    stop = False

    @staticmethod
    def learning_rate_scheduler(epoch: int):
        if epoch <= 50:
            return 1e-3
        elif 50 < epoch <= 100:
            return 5e-4
        elif 100 < epoch <= 150:
            return 1e-4
        elif 150 < epoch <= 200:
            return 5e-5
        elif 200 < epoch <= 250:
            return 1e-5
        else:
            return 1e-6
