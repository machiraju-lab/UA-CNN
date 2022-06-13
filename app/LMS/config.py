
class TCGA_SARC_Config():
    """
    Configuration for training on TCGA_SARC dataset
    """
    NAME = "SARC_LMS_256_10"

    # cross-validation config
    KFOLDS = 5

    # Training config
    EPOCHS = 100
    IMAGE_RESIZE = 256
    NUM_WORKERS = 24
    BATCH_SIZE_TRAINING = 64
    BATCH_SIZE_VALIDATION = 16
    # Using 1 to easily manage mapping between test_generator & prediction for submission preparation
    BATCH_SIZE_TESTING = 1
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    DEFAULT_LOGS_DIR = 'logs/'

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class TCGA_SARC_InferenceConfig(TCGA_SARC_Config):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
