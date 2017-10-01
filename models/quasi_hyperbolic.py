import pymc3 as pm
from models.base import Model


class QuasiHyperbolic(Model):
    @staticmethod
    def quasi_hyperbolic(t, b, d):
        return b * d ** t

    def forward(self, X, params):
        d = params['d']
        b = params['b']
        w = params['w']
        # Expected value of outcome
        u1 = self.quasi_hyperbolic(X[self.t1_col], b, d) * X[self.x1_col]
        u2 = self.quasi_hyperbolic(X[self.t2_col], b, d) * X[self.x2_col]
        return pm.math.sigmoid(w * (u2 - u1))

    def define_params(self):
        with self.model:
            self.b = pm.Uniform('b', 0, 1)
            self.d = pm.Uniform('d', 0, 1)
            self.w = pm.HalfNormal('w', sd=10)
