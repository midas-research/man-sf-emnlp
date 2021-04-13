# Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations

This codebase contains the python scripts for MAN-S, the model for the EMNLP 2020 paper [link](https://www.aclweb.org/anthology/2020.emnlp-main.676/).

## Environment & Installation Steps
Python 3.6, Pytorch, and networkx.


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset from [here](https://github.com/yumoxu/stocknet-dataset). 

Follow [link](https://tfhub.dev/google/universal-sentence-encoder/1) to generate tweet embeddings.


Generate graph

Follow [link](https://dl.acm.org/doi/10.1145/3309547) to generate the graph. 


## Run

Execute the following python command to train MAN-SF: 
```python
python train.py
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{sawhney-etal-2020-deep,
    title = "Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations",
    author = "Sawhney, Ramit  and
      Agarwal, Shivam  and
      Wadhwa, Arnav  and
      Shah, Rajiv Ratn",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.676",
    doi = "10.18653/v1/2020.emnlp-main.676",
    pages = "8415--8426"}
```

