import pymc3 as pm
from .base import Model


class ITCH(Model):
    def forward(self, X, params):
        x1, x2, t1, t2 = [X[s] for s in [self.x1_col, self.x2_col, self.t1_col, self.t2_col]]
        beta_1, beta_xA, beta_xR, beta_tA, beta_tR = \
            [params[s] for s in ['beta_1', 'beta_xA', 'beta_xR', 'beta_tA', 'beta_tR']]

        x_star = (x1 + x2) / 2.
        t_star = (t1 + t2) / 2.

        score = beta_1 + \
                beta_xA * (x2 - x1) + \
                beta_xR * (x2 - x1) / x_star + \
                beta_tA * (t2 - t1) + \
                beta_tR * (t2 - t1) / t_star

        return pm.math.sigmoid(score)

    def define_params(self):
        with self.model:
            sd = 10
            # Priors for unknown model parameters
            self.beta_1 = pm.Normal('beta_1', sd=sd)
            self.beta_xA = pm.Normal('beta_xA', sd=sd)
            self.beta_xR = pm.Normal('beta_xR', sd=sd)
            self.beta_tA = pm.Normal('beta_tA', sd=sd)
            self.beta_tR = pm.Normal('beta_tR', sd=sd)
