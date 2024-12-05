# CAP 5415 Project: A Training-free Baseline for Open-Vocabulary Video Action Detection

## 1. Requirements

Install relevant python libraries with `pip install -r requiements.txt`

Download ViCLIP-B16 weights pretrained on InternVid-10M-FLT here: (https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid) and put it in the `/viclip` folder.

## 2. Datasets

Download UCF-101 [here](https://www.crcv.ucf.edu/research/data-sets/ucf101/) and JHMDB [here](http://jhmdb.is.tue.mpg.de/). Ensure that the contents of each dataset folder look like this:

    UCF-101
    ├── ApplyMakeUp
    ├── ApplyLipstick
    ├── Archery
    ├── ...
    ├── YoYo

    JHMDB
    ├── brush_hair
    ├── catch
    ├── clap
    ├── ...
    ├── wave

## 3. Run Evaluation
Run `python baseline-py -ucf101 /path/to/ucf101 -jhmdb /path/to/jhmdb` to collect f-mAP@0.5 for UCF-101-24 and JHMDB.

## 4. Demo
I have provided a jupyter notebook in this repo to extract qualitative results. See `demo.ipynb`.