

class config:
    batch_size = 1
    epochs = 200
    steps_per_epoch = 5
    validation_steps = 4  # iteration is from zero there is actually 5 steps each val/train epoch
    log_dir = r"D:\moshe\DNN_project\log_dir2"
    model_path = r"D:\moshe\DNN_project\saved_models\Unet2"
    val_freq = 5
    stop = False

    @staticmethod
    def learning_rate_scheduler(epoch: int):
        if epoch <= 100:
            return 1e-3
        elif 100 < epoch <= 160:
            return 1e-4

        elif 160 < epoch <= 200:
            return 5e-5
        """
        else:
            return 1e-6
        """
