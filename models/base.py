import theano
import pymc3 as pm
import abc
import pandas as pd
from scipy import optimize


class Model(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, input_data, params):
        raise NotImplementedError

    def fit_MAP(self):
        try:
            MAP = pm.find_MAP(model=self.model, disp=False)
        except ValueError as e:
            if e.args[0].startswith("Optimization error"):
                print(e)
                print("Error fitting, trying again with powell method:")
                MAP = pm.find_MAP(model=self.model, disp=False, fmin=optimize.fmin_powell)
            else:
                raise e
        MAP_with_transformed = self._add_transformed_params(MAP)
        self.MAP = MAP_with_transformed
        return self.MAP

    @abc.abstractmethod
    def define_params(self):
        raise NotImplementedError

    def _add_transformed_params(self, params):
        new_params = {}
        for k, v in params.items():
            symbolic_param = self.model.named_vars[k]
            if isinstance(symbolic_param.distribution, pm.distributions.transforms.TransformedDistribution):
                base_name = "_".join(k.split("_")[:-3])
                base_param = self.model.named_vars[base_name]
                new_params[base_name] = theano.function([], base_param.transformation.backward(v))()
            new_params[k] = v
        return new_params

    def transform_prob(self, X):
        return pd.Series(theano.function([], self.forward(X, self.MAP))()).to_frame("prob")

    def transform(self, X, thresh=.5):
        return pd.Series(self.transform_prob(X) > thresh).to_frame("pred")

    def __init__(self, df, x1_col='x1', x2_col='x2', t1_col='t1', t2_col='t2', ll_col="LL"):
        self.model = pm.Model()
        self.MAP = None
        self.define_params()
        self.x1_col = x1_col
        self.x2_col = x2_col
        self.t1_col = t1_col
        self.t2_col = t2_col
        self.ll_col = ll_col
        self.train_df = df # TODO remove this

        with self.model:
            p = self.forward(df, self.model.named_vars)
            # Likelihood (sampling distribution) of observations
            likelyhood = pm.Bernoulli('Y_obs', p=p, observed=df[ll_col])
