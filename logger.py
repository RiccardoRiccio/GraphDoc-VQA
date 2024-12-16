from torch.utils.tensorboard import SummaryWriter
import datetime
import os


class Logger:
    def __init__(self, config):
        # Create log folder
        self.log_folder = config['save_dir']
        os.makedirs(self.log_folder, exist_ok=True)

        # Generate experiment name based on model name and timestamp
        experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        self.experiment_name = f"{config['model_name']}__{experiment_date}"
        self.writer = SummaryWriter(log_dir=self.log_folder)

        # Print and store configuration for reference
        self._print_config(config)

        self.current_epoch = 0
        self.len_dataset = 0  # Set dynamically based on dataset length

    def _print_config(self, config):
        """Print the configuration settings."""
        print(f"Experiment: {self.experiment_name}")
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("\n")

    def log_model_parameters(self, model):
        """Log the total and trainable model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.writer.add_text(
            "Model Info",
            f"Total Parameters: {total_params}\nTrainable Parameters: {trainable_params}"
        )
       #print(f"Model parameters: {total_params} - Trainable: {trainable_params}")

    def log_train_metrics(self, loss, lr, step):
        """Log training metrics at each batch."""
        self.writer.add_scalar("Train/Loss", loss, step)
        self.writer.add_scalar("Train/Learning Rate", lr, step)

    def log_val_metrics(self, accuracy, anls, epoch):
        """Log validation metrics at the end of each epoch."""
        self.writer.add_scalar("Val/Accuracy", accuracy, epoch)
        self.writer.add_scalar("Val/ANLS", anls, epoch)

        print(f"Epoch {epoch}: Val Accuracy: {accuracy:.4f}, Val ANLS: {anls:.4f}")

    def close(self):
        """Close the writer."""
        self.writer.close()
