# Training

## Overview

We have put a large emphasis on making training as fast as possible.
Consequently, some pre-processing steps are required.

Namely, before starting any training, we

1. Encode training audios into spectrograms and then with VAE into mean/std
2. Extract CLIP and synchronization features from videos
3. Extract CLIP features from text (captions)
4. Encode all extracted features into [MemoryMappedTensors](https://pytorch.org/tensordict/main/reference/generated/tensordict.MemoryMappedTensor.html) with [TensorDict](https://pytorch.org/tensordict/main/reference/tensordict.html)

**NOTE:** for maximum training speed (e.g., when training the base model with 2*H100s), you would need around 3~5 GB/s of random read speed. Spinning disks would not be able to catch up and most consumer-grade SSDs would struggle. In my experience, the best bet is to have a large enough system memory such that the OS can cache the data. This way, the data is read from RAM instead of disk.

The current training script does not support `_v2` training.

## Prerequisites

Install [av-benchmark](https://github.com/hkchengrex/av-benchmark). We use this library to automatically evaluate on the validation set during training, and on the test set after training.
Extract features for evaluation using [av-benchmark](https://github.com/hkchengrex/av-benchmark) for the validation and test set as a [validation cache](https://github.com/hkchengrex/MMAudio/blob/34bf089fdd2e457cd5ef33be96c0e1c8a0412476/config/data/base.yaml#L38) and a [test cache](https://github.com/hkchengrex/MMAudio/blob/34bf089fdd2e457cd5ef33be96c0e1c8a0412476/config/data/base.yaml#L31). You can also download the precomputed evaluation cache [here](https://huggingface.co/datasets/hkchengrex/MMAudio-precomputed-results/tree/main).

You will also need ffmpeg for video frames extraction. Note that `torchaudio` imposes a maximum version limit (`ffmpeg<7`). You can install it as follows:

```bash
conda install -c conda-forge 'ffmpeg<7'
```

Download the corresponding VAE (`v1-16.pth` for 16kHz training, and `v1-44.pth` for 44.1kHz training), vocoder models (`best_netG.pt` for 16kHz training; the vocoder for 44.1kHz training will be downloaded automatically), the [empty string encoding](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth), and Synchformer weights from [MODELS.md](https://github.com/hkchengrex/MMAudio/blob/main/docs/MODELS.md) place them in `ext_weights/`.

## Preparing Audio-Video-Text Features

We have prepared some example data in `training/example_videos`.
`training/extract_video_training_latents.py` extracts audio, video, and text features and save them as a `TensorDict` with a `.tsv` file containing metadata to `output_dir`.

To run this script, use the `torchrun` utility:

```bash
torchrun --standalone training/extract_video_training_latents.py
```

You can run this script with multiple GPUs (with `--nproc_per_node=<n>` after `--standalone` and before the script name) to speed up extraction.
Modify the definitions near the top of the script to switch between 16kHz/44.1kHz extraction.
Change the data path definitions in `data_cfg` if necessary.

Arguments:

- `latent_dir` -- where intermediate latent outputs are saved. It is safe to delete this directory afterwards.
- `output_dir` -- where TensorDict and the metadata file are saved.

Outputs produced in `output_dir`:

1. A directory named `vgg-{split}` (i.e., in the TensorDict format), containing
    a. `mean.memmap` mean values predicted by the VAE encoder (number of videos X sequence length X channel size)
    b. `std.memmap` standard deviation values predicted by the VAE encoder (number of videos X sequence length X channel size)
    c. `text_features.memmap` text features extracted from CLIP (number of videos X 77 (sequence length) X 1024)
    d. `clip_features.memmap` clip features extracted from CLIP (number of videos X 64 (8 fps) X 1024)
    e. `sync_features.memmap` synchronization features extracted from Synchformer (number of videos X 192 (24 fps) X 768)
    f. `meta.json` that contains the metadata for the above memory mappings
2. A tab-separated values file named `vgg-{split}.tsv` that contains two columns: `id` containing video file names without extension, and `label` containing corresponding text labels (i.e., captions)

## Preparing Audio-Text Features

We have prepared some example data in `training/example_audios`.

1. Run `training/partition_clips` to partition each audio file into clips (by finding start and end points; we do not save the partitioned audio onto the disk to save disk space)
2. Run `training/extract_audio_training_latents.py` to extract each clip's audio and text features and save them as a `TensorDict` with a `.tsv` file containing metadata to `output_dir`.

### Partitioning the audio files

Run

```bash
python training/partition_clips.py
```

Arguments:

- `data_dir` -- path to a directory containing the audio files (`.flac` or `.wav`)
- `output_dir` -- path to the output `.csv` file
- `start` -- optional; useful when you need to run multiple processes to speed up processing -- this defines the beginning of the chunk to be processed
- `end` -- optional; useful when you need to run multiple processes to speed up processing -- this defines the end of the chunk to be processed

### Extracting audio and text features

Run

```bash
torchrun --standalone training/extract_audio_training_latents.py
```

You can run this with multiple GPUs (with `--nproc_per_node=<n>`) to speed up extraction.
Modify the definitions near the top of the script to switch between 16kHz/44.1kHz extraction.

Arguments:

- `data_dir` -- path to a directory containing the audio files (`.flac` or `.wav`), same as the previous step
- `captions_tsv` -- path to the captions file, a tab-separated values (tsv) file at least with columns `id` and `caption`
- `clips_tsv` -- path to the clips file, generated in the last step
- `latent_dir` -- where intermediate latent outputs are saved. It is safe to delete this directory afterwards.
- `output_dir` -- where TensorDict and the metadata file are saved.

**Reference tsv files (with overlaps removed as mentioned in the paper) can be found [here](https://github.com/hkchengrex/MMAudio/releases/tag/v0.1).**
Note that these reference tsv files are the **outputs** of `extract_audio_training_latents.py`, which means the `id` column might contain duplicate entries (one per clip). You can still use it as the `captions_tsv` input though -- the script will handle duplicates gracefully.
Among these reference tsv files, `audioset_sl.tsv`, `bbcsound.tsv`, and `freesound.tsv` are subsets that are parts of WavCaps. These subsets might be smaller than the original datasets.
The Clotho data contains both the development set and the validation set.

Outputs produced in `output_dir`:

1. A directory named `vgg-{split}` (i.e., in the TensorDict format), containing
    a. `mean.memmap` mean values predicted by the VAE encoder (number of audios X sequence length X channel size)
    b. `std.memmap` standard deviation values predicted by the VAE encoder (number of audios X sequence length X channel size)
    c. `text_features.memmap` text features extracted from CLIP (number of audios X 77 (sequence length) X 1024)
    f. `meta.json` that contains the metadata for the above memory mappings
2. A tab-separated values file named `{basename(output_dir)}.tsv` that contains two columns: `id` containing audio file names without extension, and `label` containing corresponding text labels (i.e., captions)

## Training on Extracted Features

We use Distributed Data Parallel (DDP) for training.
First, specify the data path in `config/data/base.yaml`. If you used the default parameters in the scripts above to extract features for the example data, the `Example_video` and `Example_audio` items should already be correct.

To run training on the example data, use the following command:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 train.py exp_id=debug compile=False  debug=True example_train=True  batch_size=1
```

This will not train a useful model, but it will check if everything is set up correctly.

For full training on the base model with two GPUs, use the following command:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py exp_id=exp_1 model=small_16k
```

Any outputs from training will be stored in `output/<exp_id>`.

More configuration options can be found in `config/base_config.yaml` and `config/train_config.yaml`.
For the medium and large models, specify `vgg_oversample_rate` to be `3` to reduce overfitting.

## Checkpoints

Model checkpoints, including optimizer states and the latest EMA weights, are available here: https://huggingface.co/hkchengrex/MMAudio

---

Godspeed!
