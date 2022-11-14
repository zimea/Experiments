import os
from matplotlib import pyplot as plt


class ResultLogger:
    def __init__(self, workdir, trainer, losses, config, prior, diag):
        self.workdir = workdir
        self.trainer = trainer
        self.losses = losses
        self.config = config
        self.prior = prior
        self.diag = diag

        self.output_dir = os.path.abspath(os.path.join(workdir, config.plots))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.prior_means, self.prior_stds = prior.estimate_means_and_stds()

    def __runResimulations__(self):
        res = {}
        res["raw_sims"] = self.trainer.generative_model(
            batch_size=self.config.resimulation_param["simulations"]
        )
        res["validation_sims"] = self.trainer.configurator(res["raw_sims"])
        res["post_samples"] = self.trainer.amortizer.sample(
            res["validation_sims"], self.config.resimulation_param["post_samples"]
        )
        res["post_samples_unnorm"] = (
            self.prior_means + res["post_samples"] * self.prior_stds
        )
        return res

    def create_plots(self):
        plot_dir = os.path.abspath(os.path.join(self.workdir, self.config.plots))
        if self.config.losses:
            loss = self.diag.plot_losses(self.losses)
            loss.savefig(os.path.join(plot_dir, "losses.png"))
            plt.close(loss)

        if self.config.latent2d:
            latent2d = self.trainer.diagnose_latent2d()
            latent2d.savefig(os.path.join(plot_dir, "latent2d.png"))
            plt.close(latent2d)

        if self.config.sbc_histograms:
            sbc_histograms = self.trainer.diagnose_sbc_histograms()
            sbc_histograms.savefig(os.path.join(plot_dir, "sbc_histograms.png"))
            plt.close(sbc_histograms)

        if self.config.run_resimualtions:
            res = self.__runResimulations__()

        if self.config.sbc_ecdf:
            sbc_ecdf = self.diag.plot_sbc_ecdf(
                res["post_samples"], res["validation_sims"]["parameters"]
            )
            sbc_ecdf.savefig(os.path.join(plot_dir, "sbc_ecdf.png"))
            plt.close(sbc_ecdf)

        # TODO: posterior scores, correlation

        if self.config.recovery:
            recovery = self.diag.plot_recovery(
                res["post_samples"],
                res["validation_sims"]["parameters"],
                param_names=self.config.prior_names,
            )
            recovery.savefig(os.path.join(plot_dir, "recovery.png"))
            plt.close(recovery)
