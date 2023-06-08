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
#### TO DO
- get rid of spark session and pandas manipualtions in KL-UCB, i.e. perform the calculations directly on spark dataframes (this would require understanding and usage of `Spark UDF`s and `Apache Arrow` in `PySpark`)
- write `KL_UCB` class documentation
- inherit `KL_UCB` ftom `UCB` class to avoid multiple code duplicates
- play with UCB and KL-UCB exploration coefficients to improve on-MovieLens performance



# Notes

## Useful Links:
- **Replay** https://github.com/sb-ai-lab/RePlay
- **OBP**    https://github.com/st-tech/zr-obp?ysclid=li50kcw2ru470022012 
- **KL-UCB** https://arxiv.org/pdf/1102.2490.pdf
- **LinUCB** https://arxiv.org/pdf/1003.0146.pdf
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

________________________________________________________________________________
*Arkadiy Vladimirov* © *2023*