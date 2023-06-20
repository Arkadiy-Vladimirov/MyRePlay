import math

from replay.models import UCB
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.optimize import root_scalar


class KL_UCB(UCB):
    """Single actor Bernoulli bandit model. Same as UCB class it fits on a 
    dataset of items labeled as 1 for positive interaction and 0 for negative.
    The items are considered as bandits and the labels as outcomes of 
    interactions with them. The algortihm collects the statistics of the 
    interactions and estimates the true fraction of positive outcomes with an 
    upper confidence bound. The prediction (recommendation) is a set of items
    with the highest upper bounds.
    
    The exact formula of the upper bound below is a bit trickier than such of 
    the classical ucb but is `proven to give asymptotically better results 
    <https://arxiv.org/pdf/1102.2490.pdf>`_.

    .. math::
        pred_i = \\max q \\in [0,1] : 
        n_i \\cdot \\operatorname{KL}\\left(\\frac{p_i}{n_i}, q \\right) 
        \\leq \\log(n) + c \\log(\\log(n)),

    where 

    :math:`pred_i` is the upper bound (predicted relevance), 

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

    """

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
            