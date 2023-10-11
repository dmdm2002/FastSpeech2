class HyperParams:
    def __init__(self):
        self.ckpt_path = "D:/Side/BackUp/output/ckpt/US_Female"
        self.log_path = "D:/Side/BackUp/output/log/US_Female"
        self.result_path = "D:/Side/BackUp/output/result/US_Female"

        ################################
        # data Parameters             #
        ################################
        self.load_mel_from_disk = False
        self.training_files = 'D:/Side/DB/US_Female/metadata.txt'
        self.validation_files = 'D:/Side/DB/US_Female/metadata.txt'
        self.text_cleaners = ['english_cleaners']
        self.n_frames_per_step = 1  # currently only 1 is supported

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Optimization Hyperparameters #
        ################################
        self.seed = 1234

        self.batch_size = 48
        self.betas = [0.9, 0.98]
        self.eps = 0.000000001
        self.weight_decay = 0.0
        self.grad_clip_thresh = 1.0
        self.grad_acc_step = 1
        self.warm_up_step = 4000
        self.anneal_steps = []
        self.anneal_rate = 1.0
        self.device = "cuda:0"

        self.epochs = 500
        self.log_step = 10
        self.synth_step = 1000
        self.val_step = 1000
        self.save_step = 10000
