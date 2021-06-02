from functools import cmp_to_key
import numpy as np
import torch
from tqdm import tqdm
from wavenet_vocoder import builder


class Map(dict):
	"""
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    Credits to epool:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

	def __init__(self, *args, **kwargs):
		super(Map, self).__init__(*args, **kwargs)
		for arg in args:
			if isinstance(arg, dict):
				for k, v in arg.items():
					self[k] = v

		if kwargs:
			for k, v in kwargs.iteritems():
				self[k] = v

	def __getattr__(self, attr):
		return self.get(attr)

	def __setattr__(self, key, value):
		self.__setitem__(key, value)

	def __setitem__(self, key, value):
		super(Map, self).__setitem__(key, value)
		self.__dict__.update({key: value})

	def __delattr__(self, item):
		self.__delitem__(item)

	def __delitem__(self, key):
		super(Map, self).__delitem__(key)
		del self.__dict__[key]


# Default hyperparameters:
wavenet_hparams = Map({
	'name': "wavenet_vocoder",

	# Convenient model builder
	'builder': "wavenet",

	# Input type:
	# 1. raw [-1, 1]
	# 2. mulaw [-1, 1]
	# 3. mulaw-quantize [0, mu]
	# If input_type is raw or mulaw, network assumes scalar input and
	# discretized mixture of logistic distributions output, otherwise one-hot
	# input and softmax output are assumed.
	# **NOTE**: if you change the one of the two parameters below, you need to
	# re-run preprocessing before training.
	'input_type': "raw",
	'quantize_channels': 65536,  # 65536 or 256

	# Audio:
	'sample_rate': 16000,
	# this is only valid for mulaw is True
	'silence_threshold': 2,
	'num_mels': 80,
	'fmin': 125,
	'fmax': 7600,
	'fft_size': 1024,
	# shift can be specified by either hop_size or frame_shift_ms
	'hop_size': 256,
	'frame_shift_ms': None,
	'min_level_db': -100,
	'ref_level_db': 20,
	# whether to rescale waveform or not.
	# Let x is an input waveform, rescaled waveform y is given by:
	# y = x / np.abs(x).max() * rescaling_max
	'rescaling': True,
	'rescaling_max': 0.999,
	# mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
	# happen depends on min_level_db and ref_level_db, causing clipping noise.
	# If False, assertion is added to ensure no clipping happens.o0
	'allow_clipping_in_normalization': True,

	# Mixture of logistic distributions:
	'log_scale_min': float(-32.23619130191664),

	# Model:
	# This should equal to `quantize_channels` if mu-law quantize enabled
	# otherwise num_mixture * 3 (pi, mean, log_scale)
	'out_channels': 10 * 3,
	'layers': 24,
	'stacks': 4,
	'residual_channels': 512,
	'gate_channels': 512,  # split into 2 gropus internally for gated activation
	'skip_out_channels': 256,
	'dropout': 1 - 0.95,
	'kernel_size': 3,
	# If True, apply weight normalization as same as DeepVoice3
	'weight_normalization': True,
	# Use legacy code or not. Default is True since we already provided a model
	# based on the legacy code that can generate high-quality audio.
	# Ref: https://github.com/r9y9/wavenet_vocoder/pull/73
	'legacy': True,

	# Local conditioning (set negative value to disable))
	'cin_channels': 80,
	# If True, use transposed convolutions to upsample conditional features,
	# otherwise repeat features to adjust time resolution
	'upsample_conditional_features': True,
	# should np.prod(upsample_scales) == hop_size
	'upsample_scales': [4, 4, 4, 4],
	# Freq axis kernel size for upsampling network
	'freq_axis_kernel_size': 3,

	# Global conditioning (set negative value to disable)
	# currently limited for speaker embedding
	# this should only be enabled for multi-speaker dataset
	'gin_channels': -1,  # i.e., speaker embedding dim
	'n_speakers': -1,

	# Data loader
	'pin_memory': True,
	'num_workers': 2,

	# train/test
	# test size can be specified as portion or num samples
	'test_size': 0.0441,  # 50 for CMU ARCTIC single speaker
	'test_num_samples': None,
	'random_state': 1234,

	# Loss

	# Training:
	'batch_size': 2,
	'adam_beta1': 0.9,
	'adam_beta2': 0.999,
	'adam_eps': 1e-8,
	'amsgrad': False,
	'initial_learning_rate': 1e-3,
	# see lrschedule.py for available lr_schedule
	'lr_schedule': "noam_learning_rate_decay",
	'lr_schedule_kwargs': {},  # {"anneal_rate": 0.5, "anneal_interval": 50000},
	'nepochs': 2000,
	'weight_decay': 0.0,
	'clip_thresh': -1,
	# max time steps can either be specified as sec or steps
	# if both are None, then full audio samples are used in a batch
	'max_time_sec': None,
	'max_time_steps': 8000,
	# Hold moving averaged parameters and use them for evaluation
	'exponential_moving_average': True,
	# averaged = decay * averaged + (1 - decay) * x
	'ema_decay': 0.9999,

	# Save
	# per-step intervals
	'checkpoint_interval': 10000,
	'train_eval_interval': 10000,
	# per-epoch interval
	'test_eval_epoch_interval': 5,
	'save_optimizer_state': True,

	# Eval:
})


class Synthesizer(object):

    def __init__(self, device):
        
        self.model = getattr(builder, wavenet_hparams.builder)(
            out_channels=wavenet_hparams.out_channels,
            layers=wavenet_hparams.layers,
            stacks=wavenet_hparams.stacks,
            residual_channels=wavenet_hparams.residual_channels,
            gate_channels=wavenet_hparams.gate_channels,
            skip_out_channels=wavenet_hparams.skip_out_channels,
            cin_channels=wavenet_hparams.cin_channels,
            gin_channels=wavenet_hparams.gin_channels,
            weight_normalization=wavenet_hparams.weight_normalization,
            n_speakers=wavenet_hparams.n_speakers,
            dropout=wavenet_hparams.dropout,
            kernel_size=wavenet_hparams.kernel_size,
            upsample_conditional_features=wavenet_hparams.upsample_conditional_features,
            upsample_scales=wavenet_hparams.upsample_scales,
            freq_axis_kernel_size=wavenet_hparams.freq_axis_kernel_size,
            scalar_input=True,
            legacy=wavenet_hparams.legacy,
        )
        self.device = device

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])

    def spect2wav(self, c=None, tqdm=tqdm):
        self.model.eval()
        self.model.make_generation_fast_()

        Tc = c.shape[0]
        upsample_factor = wavenet_hparams.hop_size
        # Overwrite length according to feature size
        length = Tc * upsample_factor

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

        initial_input = torch.zeros(1, 1, 1).fill_(0.0)

        # Transform data to GPU
        initial_input = initial_input.to(self.device)
        c = None if c is None else c.to(self.device)

        with torch.no_grad():
            y_hat = self.model.incremental_forward(
                initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=wavenet_hparams.log_scale_min)

        y_hat = y_hat.view(-1).cpu().data.numpy()

        return y_hat

    def batch_spect2wav(self, c=None, tqdm=tqdm):
        self.model.eval()
        self.model.make_generation_fast_()

        B = c.size(0)
        Tc = c.size(1)
        upsample_factor = wavenet_hparams.hop_size
        # Overwrite length according to feature size
        length = Tc * upsample_factor

        # B x C x T
        c = torch.FloatTensor(c.permute(0, 2, 1))

        initial_input = torch.zeros(B, 1, 1).fill_(0.0)

        # Transform data to GPU
        initial_input = initial_input.to(self.device)
        c = None if c is None else c.to(self.device)

        with torch.no_grad():
            y_hat = self.model.incremental_forward(
                initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=wavenet_hparams.log_scale_min)

        y_hat = y_hat.view(B, -1).cpu().data.numpy()

        return y_hat

    def file2wav(self, fname):
        spect = np.load(fname)
        return self.spect2wav(c=spect)
