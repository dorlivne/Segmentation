

class config:
    batch_size = 8
    epochs = 100
    steps_per_epoch = 4
    validation_steps = 4  # iteration is from zero there is actually 5 steps each val/train epoch
    log_dir = r"log_dir"
    model_path = r"saved_models/Unet"
    val_freq = 2

    @staticmethod
    def learning_rate_schedulere(epoch: int):
        if epoch <= 5:
            return 1e-3
        elif 5 < epoch <= 10:
            return 1e-4
        elif 10 < epoch <= 15:
            return 1e-5
        elif 15 < epoch <= 20:
            return 5e-6
        else:
            return 1e-6
