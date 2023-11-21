# Overview

Based on the original CPA-LGC (https://github.com/jindeok/CPA-LGC-Recbole) architecture and implementation for the paper  
Jin-Duk Park, Siqing Li, Won-Yong Shin, and Xin Cao,
"Criteria Tell You More than Ratings: Criteria Preference-Aware Light Graph Convolution for Effective Multi-Criteria Recommendation",  
Proceedings of the 29th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, **KDD '23**

# Installation

Run

```
pip install -r requirements.txt
```

# Dataset

YM/multi_YM.csv contains the original Yahoo Movies Dataset.  
The MC expansion graph datasets are formed by running preprocess.py (YM.tr.inter: training set, YM.ts.inter: test dataset, YM.val.inter: validation set, YM.inter: original dataset)

# How to Use

1. Run preprocess.py on the multi_YM.csv dataset to split your dataset into the training set, validation set, and test set
2. After preprocessing the data, run main.py to train and evaluate the model

## Error Handling

If you run into this error while running main.py,

```
Traceback (most recent call last):
  File "/Users/danieljo/Multi-criteria-Recommend-System/main.py", line 107, in <module>
    main()
  File "/Users/danieljo/Multi-criteria-Recommend-System/main.py", line 98, in main
    results = trainer.evaluate(test_data)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieljo/Multi-criteria-Recommend-System/venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieljo/Multi-criteria-Recommend-System/venv/lib/python3.11/site-packages/recbole/trainer/trainer.py", line 626, in evaluate
    result = self.evaluator.evaluate(struct)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieljo/Multi-criteria-Recommend-System/venv/lib/python3.11/site-packages/recbole/evaluator/evaluator.py", line 39, in evaluate
    metric_val = self.metric_class[metric].calculate_metric(dataobject)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieljo/Multi-criteria-Recommend-System/venv/lib/python3.11/site-packages/recbole/evaluator/metrics.py", line 182, in calculate_metric
    result = self.metric_info(pos_index, pos_len)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieljo/Multi-criteria-Recommend-System/venv/lib/python3.11/site-packages/recbole/evaluator/metrics.py", line 190, in metric_info
    iranks = np.zeros_like(pos_index, dtype=np.float)
                                            ^^^^^^^^
  File "/Users/danieljo/Multi-criteria-Recommend-System/venv/lib/python3.11/site-packages/numpy/__init__.py", line 319, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'cfloat'?
```

Go into the RecBole code and change these lines of the metric_info function of the NDCG class into the following:

```
iranks = np.zeros_like(pos_index, dtype=np.cfloat)
ranks = np.zeros_like(pos_index, dtype=np.cfloat)
```
