import sys, os
import config

sys.path.append(os.path.abspath(os.path.join(config.BayesFlowPath)))
import bayesflow.diagnostics as diag
from bayesflow.forward_inference import Prior


class ResultLogger:
    def __init__(self, trainer, losses, workdir):
        self.workdir = workdir
        self.trainer = trainer
        self.losses = losses

    def __runResimulations__(self):
        res = {}
        res["raw_sims"] = self.trainer.generative_model(
            batch_size=config.resimulation_param["simulations"]
        )
        res["validation_sims"] = self.trainer.configurator(res["raw_sims"])
        res["post_samples"] = self.trainer.amortizer.sample(
            res["validation_sims"], config.resimulation_param["post_sampels"]
        )
        prior = Prior(prior_fun=config.prior_func, param_names=config.prior_names)
        prior_means, prior_stds = prior.estimate_means_and_stds()
        res["post_samples_unnorm"] = prior_means + res["post_samples"] * prior_stds
        return res

    def create_plots(self):
        if config.losses:
            loss = diag.plot_losses(self.losses)
            loss.savefig(os.path.join(self.workdir, "losses.png"))

        if config.latent2d:
            latent2d = self.trainer.diagnose_latent2d()
            latent2d.savefig(os.path.join(self.workdir, "latent2d.png"))

        if config.sbc_histograms:
            sbc_histograms = self.trainer.diagnose_sbc_histograms()
            sbc_histograms.savefig(os.path.join(self.workdir, "sbc_histograms.png"))

        if config.run_resimualtions:
            res = self.__runResimulations__()

        if config.sbc_ecdf:
            sbc_ecdf = diag.plot_sbc_ecdf(
                res["post_samples"], res["validation_sims"]["parameters"]
            )
            sbc_ecdf.savefig(os.path.join(self.workdir, "sbc_ecdf.png"))

        # TODO: posterior scores, correlation

        if config.recovery:
            recovery = diag.plot_recovery(
                res["post_samples"],
                res["validation_sims"]["parameters"],
                param_names=config.prior_names,
            )
            recovery.savefig(os.path.join(self.workdir, "recovery.png"))
