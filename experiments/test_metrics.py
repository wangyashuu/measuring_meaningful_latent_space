import numpy as np
from disentangling.metrics import (
    z_min_var,
    mig,
    mig_sup,
    dcimig,
    sap,
    modularity,
    dci,
    dcii,
)
from disentangling.metrics.utils import get_scores
import wandb

from experiments.utils.seed_everything import seed_everything


class Logger:
    def log(self, data, step=None):
        self.run.log(data, step=step)

    def start(self, config):
        self.run = wandb.init(project="metric_eval", config=config)

    def finish(self):
        self.run.finish()


def test_metrics(n_times):
    metrics = [dcii]
    cases = [
        [False, False, False],
        [False, False, True],
        [False, True, False],
        [False, True, True],
        [True, False, False],
        [True, False, True],
        [True, True, False],
        [True, True, True],
    ]
    logger = Logger()
    for i in range(n_times):
        seed_everything(i)
        for c in cases:
            # c_name = ''.join((np.array(c).astype(int).astype(str)))
            logger.start(config=dict(case=c, seed=i))
            all_results = {}
            for fn in metrics:
                res = get_scores(fn, *c)
                all_results.update(
                    {f"{fn.__name__}__{k}": res[k] for k in res}
                    if type(res) is dict
                    else {fn.__name__: res}
                )
            logger.log(all_results, step=i)
            logger.finish()
    print("end")


test_metrics(50)