# ePiC

This repository contains the PyTorch code for models implemented in our ACL 2022 paper, ["ePiC: Employing Proverbs in Context as a Benchmark for Abstract Language Understanding"](https://arxiv.org/abs/2109.06838) 

Benchmark website: [epic-benchmark.github.io](https://epic-benchmark.github.io/)

## Installation:

Setup an anaconda environment (recommended with python 3.8.8). Following this install the packages mentioned in `requirements.txt` as
```
pip install -r requirements.txt
```

## Download dataset
```
bash download_dataset.sh
```
The training and evaluation splits for tasks are provided in the data folder

## Training and Evaluation 

#### Proverb and Alignment prediction

+ Proverb Prediction
```
python src/proverb_prediction.py # for seen proverbs
python src/proverb_prediction_unseen.py # for unseen proverbs
```

+ Alignment prediction
```
python src/span_prediction.py
```

+ Joint proverb and alignment prediction
```
python src/joint_proverb_span_prediction.py
```

#### Narrative Generation
```
python src/bart_narrative_generator.py
python src/t5_narrative_generator.py
```


#### Identifying narratives with similar motifs
```
python src/identify_similar_narrative.py
```


## Citation

```
@inproceedings{
    title={e{P}i{C}: Employing Proverbs in Context as a Benchmark for Abstract Language Understanding},
    author={Ghosh, Sayan and Srivastava, Shashank},
    booktitle={ACL},
    year={2022}
}
```