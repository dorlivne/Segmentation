

class config:
    # some parameters and thresholds
    batch_size = 1
    epochs = 400
    steps_per_epoch = 4
    validation_steps = 4  # iteration is from zero there is actually 5 steps each val/train epoch
    log_dir = r"log_dir"
    model_path = r"saved_models/Unet"
    val_freq = 2
    stop = False
    alpha = 16
    sigma = 4
    converged_threshold = 30
    Augment = False
    elastic_threshold = 0.0
    mirror_threshold = 0.0
    use_assaf = True
    visual = False

    @staticmethod
    def learning_rate_scheduler(epoch: int):
        if epoch <= 20:
            return 1e-3
        elif 20 < epoch <= 50:
            return 5e-4
        elif 50 < epoch <= 100:
            return 1e-4
        elif 100 < epoch <= 200:
            return 5e-5
        elif 200 < epoch <= 250:
            return 1e-5
        else:
            return 5e-6
