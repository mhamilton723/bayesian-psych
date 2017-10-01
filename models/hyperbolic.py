import pymc3 as pm
from models.base import Model

class Hyperbolic(Model):
    @staticmethod
    def hyperbolic(t, k):
        return 1. / (1. + (k * t))

    def forward(self, X, params):
        k = params['k']
        w = params['w']
        # Expected value of outcome
        u1 = self.hyperbolic(X[self.t1_col], k) * X[self.x1_col]
        u2 = self.hyperbolic(X[self.t2_col], k) * X[self.x2_col]
        return pm.math.sigmoid(w * (u2 - u1))

    def define_params(self):
        with self.model:
            self.k = pm.Normal('k', sd=10)
            self.w = pm.HalfNormal('w', sd=10)


