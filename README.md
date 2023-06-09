# My RePlay

This is my private copy of sb-ai-lab/RePlay repository (08.08.2023 version). 

Here I perform experiments, write notes and such. The original Replay README.md 
is RePlayREADME.md here.

# Task List
### KL-UCB:
#### DONE
- implement KL-UCB algorithm
- test KL-UCB in `rlberry` framework
- inject KL-UCB into `UCB` class and obtain metrics on MovieLens
- put the KL-UCB into a separate `KL_UCB` class
- get rid of spark session parameter in KL-UCB constructor
- get rid of pandas manipualtions in KL-UCB, i.e. perform the calculations directly on spark dataframes (via PySpark UDF)
- inherit `KL_UCB` ftom `UCB` class to avoid multiple code duplicates
- finish `KL_UCB` class documentation 
- change default `exploration_coef` parameter in `KL_UCB.__init__()`
- commit `KL_UCB` to HDI Lab's Replay repo
#### TO DO
- cover `KL_UCB` with unit tests
- docs/pages/useful_data/algorithm_selection.md: add KL-UCB to Model Comparison
- docs/pages/installation.rst: add KL-UCB to model_inheritance image
- add obp to poetry toml

### HCB:
#### DONE
- tree constructor prototype 
#### TO DO
- tree fit protoype



# Notes

## Useful Links:
- [**Replay**](https://github.com/sb-ai-lab/RePlay)
- [**Open Bandit Pipeline**](https://github.com/st-tech/zr-obp?ysclid=li50kcw2ru470022012)
- [**KL-UCB**](https://arxiv.org/pdf/1102.2490.pdf)
- [**LinUCB**](https://arxiv.org/pdf/1003.0146.pdf)
________________________________________________________________________________

## Installation guide:

Assume we are at the root directory of the project.
- Install `poetry`
```
$ pip install poetry
```
- Set poetry `virtualenvs.in-project` as `true`
```
$ poetry config virtualenvs.in-project true
```
This will force poetry to 
create local virtual enviroment `.venv` in project root directory.

- Install dependencies by `poetry install`
```
$ poetry install
```
- Now we can build the project with `poetry build`
```
$ poetry build
```
The result is a wheel file `replay_rec-0.10.0-py3-none-any.whl` 
at `dist` folder that may be directly installed as a python package.
- To install the obatained package in `.venv` activate the environment by 
```
$ source ./.venv/bin/activate
```
- And install the replay itself
```
$ pip install --force-reinstall dist/replay_rec-0.10.0-py3-none-any.whl
```
Now we may peacefully run notebooks at `./experiments`. (don't forget to select 
recently made `.venv` kernel in your notebook!)
- You may also use
```
$ bash build.sh build
```
inside `.venv` environment to build the project with docs inside `./build` 
directory. 

________________________________________________________________________________

## UCB
The implemented UCB requires interaction log of pairs (item_idx, 0/1) with 0 
for "negative" and 1 for "positive" interaction and operates as a UCB algorithm 
in a single (all users are seen as one aggregated agent) k-armed Bernoulli 
bandit problem. 
________________________________________________________________________________

## KL-UCB
The KL-UCB implementation may be obtained with a slight modification of 
`ucb.py` module. Particularly, all statistics needed are the same and are 
gathered during `_fit`, `_refit` methods, but the relevance has to be computed 
with a different formula, i.e. (in `ucb.py` notation)
```
    relevance = max q, such that
    total * d(pos/total, q) <= log(full_count) + coef * log(log(full_count)), (1)
    0 <= q <= 1,
```
where `d(p,q)` denotes KL-divergence between p- and q- Bernoulli distributions. 
The closed form may be written as
```
d(p,q) = p * log(p/q) + (1-p) * log((1-p)/(1-q)).
```
As the **KL-UCB** article suggests the optimization problem (1) may be 
efficiently computed using Newton iterations. Although the thoretical bounds 
are proven for `coef` = 3, authors also recommend to take `coef` = 0 for 
optimal performance in practice.
________________________________________________________________________________

## PySpark
- on `pandas_udf` vs `udf`: [link](https://www.databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html) (VPN required)
- on `pandas` vectorization: [link](https://pythonspeed.com/articles/pandas-vectorization/)

## Docs
- [Math support in Sphinx](https://sphinx-experiment.readthedocs.io/en/latest/ext/math.html)
- [How to write docs](https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/)

## Contributing
- [contribution guidelines](https://github.com/sb-ai-lab/RePlay/blob/main/CONTRIBUTING.md)
________________________________________________________________________________
*Arkadiy Vladimirov* © *2023*
