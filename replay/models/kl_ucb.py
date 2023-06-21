import math

from typing import Optional
from replay.models import UCB
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.optimize import root_scalar

class KL_UCB(UCB):
    """
    Single actor Bernoulli `bandit model 
    <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_. Same to :class:`UCB` 
    computes item relevance as an upper confidence bound of true fraction of 
    positive interactions.

    In a nutshell, KL-UCB сonsiders the data as the history of interactions with 
    items. The interaction may be either positive or negative. For each item 
    the model computes the empirical frequency of positive interactions and 
    estimates the true frequency with an upper confidence bound. The higher 
    the bound for an item is the more relevant it is presumed.

    The upper bound below is what differs from the classical UCB. It is 
    computed according to the `original article 
    <https://arxiv.org/pdf/1102.2490.pdf>`_ where is proven to produce 
    assymptotically better results.

    .. math::
        u_i = \\max q \\in [0,1] : 
        n_i \\cdot \\operatorname{KL}\\left(\\frac{p_i}{n_i}, q \\right) 
        \\leqslant \\log(n) + c \\log(\\log(n)),

    where 

    :math:`u_i` -- upper bound for item :math:`i`, 

    :math:`c` -- exploration coeficient,

    :math:`n` -- number of interactions in log,

    :math:`n_i` -- number of interactions with item :math:`i`,

    :math:`p_i` -- number of positive interactions with item :math:`i`,

    and

    .. math::
        \\operatorname{KL}(p, q) 
        = p \\log\\frac{p}{q} + (1-p)\\log\\frac{1-p}{1-q}
    
    is the KL-divergence of Bernoulli distribution with parameter :math:`p` 
    from Bernoulli distribution with parameter :math:`q`.

    Being a bit trickier though, the bound shares with UCB the same 
    exploration-exploitation tradeoff dilemma. You may increase the `c` 
    coefficient in order to shift the tradeoff towards exploration or decrease 
    it to set the model to be more sceptical of items with small volume of 
    collected statistics. The authors of the `article 
    <https://arxiv.org/pdf/1102.2490.pdf>`_ though claim `c = 0` to be of the 
    best choice in practice.

    
    As any other RePlay model, KL-UCB takes a log to fit on as a ``DataFrame`` 
    with columns ``[user_idx, item_idx, timestamp, relevance]``. Following the 
    procedure above, KL-UCB would see each row as a record of an interaction 
    with ``item_idx`` with positive (relevance = 1) or negative (relevance = 0) 
    outcome. ``user_idx`` and ``timestamp`` are ignored. 
    
    If ``relevance`` column is not of 0/1 initially, then you have to decide 
    what kind of relevance has to be considered as positive and convert 
    ``relevance`` to binary format during preprocessing.

    To provide a prediction, KL-UCB would sample a set of recommended items for 
    each user with probabilites proportional to obtained relevances.
    
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2, 3, 3], "item_idx": [1, 2, 1, 2], "relevance": [1, 0, 0, 0]})
    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = KL_UCB()
    >>> model.fit(data_frame)
    >>> model.predict(data_frame,k=2,users=[1,2,3,4], items=[1,2,3]
    ... ).toPandas().sort_values(["user_idx","relevance","item_idx"],
    ... ascending=[True,False,True]).reset_index(drop=True)
       user_idx  item_idx  relevance
    0         1         3   1.000000
    1         1	        2   0.750000
    2         2	        3   1.000000
    3         2	        1   0.933013
    4         3	        3   1.000000
    5         4	        3   1.000000
    6         4	        1   0.933013

    """

    def __init__(
    self,
    exploration_coef: float = 0.0,
    sample: bool = False,
    seed: Optional[int] = None,
    ):
        """
        :param exploration_coef: exploration coefficient
        :param sample: flag to choose recommendation strategy.
            If True, items are sampled with a probability proportional
            to the calculated predicted relevance.
            Could be changed after model training by setting the `sample` 
            attribute.
        :param seed: random seed. Provides reproducibility if fixed
        """

        super().__init__(exploration_coef, sample, seed)

    def _calc_item_popularity(self):

        rhs = math.log(self.full_count) \
            + self.coef * math.log(math.log(self.full_count))
        eps = 1e-12

        def Bernoulli_KL(p,q) :
            return p * math.log(p/q) + (1-p) * math.log((1-p)/(1-q))

        @udf(returnType=DoubleType())
        def get_ucb(pos, total) :
            p = pos / total
            
            if (p == 0) :
                ucb = root_scalar(  
                    f = lambda q: math.log(1/(1-q)) - rhs,
                    bracket = [0, 1-eps],
                    method = 'brentq').root
                return ucb
            
            if (p == 1) :
                ucb = root_scalar(
                    f = lambda q: math.log(1/q) - rhs,
                    bracket = [0+eps, 1],
                    method = 'brentq').root
                return ucb
            
            ucb = root_scalar(
                    f = lambda q: total * Bernoulli_KL(p, q) - rhs,
                    bracket = [p, 1-eps],
                    method = 'brentq').root
            return ucb
            
        items_counts = self.items_counts_aggr.withColumn(
            "relevance", get_ucb("pos", "total")
        )

        self.item_popularity = items_counts.drop("pos", "total")

        self.item_popularity.cache().count()
        self.fill = 1 + math.sqrt(self.coef * math.log(self.full_count))
            