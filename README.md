# DCASE2024 Task 9 Baseline System

The code of this repository is mostly from [AudioSep](https://github.com/Audio-AGI/AudioSep). If you use this code, please cite the original repository following:

```bibtex
@article{liu2023separate,
  title={Separate Anything You Describe},
  author={Liu, Xubo and Kong, Qiuqiang and Zhao, Yan and Liu, Haohe and Yuan, Yi, and Liu, Yuzhuo, and Xia, Rui and Wang, Yuxuan, and Plumbley, Mark D and Wang, Wenwu},
  journal={arXiv preprint arXiv:2308.05037},
  year={2023}
}
```
```bibtex
@inproceedings{liu22w_interspeech,
  title={Separate What You Describe: Language-Queried Audio Source Separation},
  author={Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi, and Plumbley, Mark D and Wang, Wenwu},
  year=2022,
  booktitle={Proc. Interspeech},
  pages={1801--1805},
}
```

## Setup
Clone the repository and setup the conda environment: 

  ```shell
  git clone https://github.com/Audio-AGI/dcase2024_task9_baseline.git && \
  cd dcase2024_task9_baseline && \ 
  conda env create -f environment.yml && \
  conda activate AudioSep
  ```
Download [CLAP](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint) (music_speech_audioset_epoch_15_esc_89.98.pt
) model weight at `checkpoint/`, which is necessary for training AudioSep model.
<hr>

## Training 

To utilize your audio-text paired dataset:

1. Format your dataset to match our JSON structure. Refer to the provided template at `datafiles/template.json`.

2. Update the `config/audiosep_base.yaml` file by listing your formatted JSON data files under `datafiles`. For example:

```yaml
data:
    datafiles:
        - 'datafiles/your_datafile_1.json'
        - 'datafiles/your_datafile_2.json'
        ...
```

Train AudioSep from scratch:
  ```python
  python train.py --workspace workspace/AudioSep --config_yaml config/audiosep_base.yaml --resume_checkpoint_path ''
  ```

Finetune AudioSep from pretrained checkpoint:
  ```python
  python train.py --workspace workspace/AudioSep --config_yaml config/audiosep_base.yaml --resume_checkpoint_path path_to_checkpoint
  ```

The development set including Clotho v2 and augmented FSD50k datasets can be found [here](https://zenodo.org/records/10887496).

## Baseline model
For the baseline model, we trained the AudioSep model using Clotho and augmented FSD50K datasets for 200k steps with a batch size of 16 using one Nvidia A100 GPU (around 1 day). The checkpoint of the baseline model can be found [here](https://zenodo.org/records/10887460).
<hr>

## Evaluation
* Download the audio files and the indexes of the synthetic validation data from [Zenodo](https://zenodo.org/records/10886481). 

```yaml
# paired data indexes for evaluation
- lass_synthetic_validation.csv
# validation audio files
- lass_validation:
    - audio_file_1.wav
    ...
```



* Modify the paths as shown below and then run the evaluation script `dcase_evaluator.py`. 

```python
dcase_evaluator = DCASEEvaluator(
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation'
    )

checkpoint_path = 'audiosep,baseline-16k,step=200000.ckpt'
eval(dcase_evaluator, checkpoint_path)

"""
-------  Start Evaluation  -------
Evaluation on DCASE T9 synthetic validation set.
Results: SDR: SDR: 5.708, SDRi: 5.673, SISDR: 3.862
------------- Done ---------------
"""
```



