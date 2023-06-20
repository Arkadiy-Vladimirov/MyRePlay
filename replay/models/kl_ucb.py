import math

from replay.models import UCB
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.optimize import root_scalar


class KL_UCB(UCB):
    """Single actor Bernoulli bandit model. Same as UCB class it takes the 
    dataset of items labeled as 1 for positive interaction and 0 for negative.
    The items are considered as bandits and the labels as outcomes of 
    interactions with them. The algortihm collects the statistics of the 
    interactions and estimates the true fraction of positive outcomes with an 
    upper confidence bound. The prediction (recommendation) is a set of items
    with the highest upper bounds.
    
    The exact formula of upper bound is a bit tricker than such of the 
    classical ucb and is proved to give asymptotically better results.

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
            