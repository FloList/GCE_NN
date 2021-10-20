"""
This script computes constraints on the Poissonian flux fraction using the neural network estimator
discussed in arXiv:2107.09070.
"""
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.losses import Loss
from sklearn.model_selection import train_test_split
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from matplotlib.ticker import MultipleLocator, LogLocator
import os
import copy
import seaborn as sns
import tensorflow as tf
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)

# ######################################################################################################################
if __name__ == '__main__':
    # ########################################################
    NN_TYPE = "CNN"  # "CNN" or "U-Net"
    GADI = True  # run on Gadi?
    DEBUG = False  # debug mode (verbose and with plots)?
    PRE_GEN = True  # use pre-generated data (CNN only)
    TASK = "TEST"  # "TRAIN" or "TEST"
    RESUME = False  # resume training? WARNING: if False, content of summary and checkpoint folders will be deleted!
    # Options for testing
    TEST_CHECKPOINT = None  # string with global time step to restore. if None: restore latest checkpoint
    TEST_EXP_PATH = "./checkpoints/Fermi_example_add_two_256_BN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    # Fix random seeds
    np.random.seed(0)
    tf.random.set_seed(0)

    # Plot settings
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rcParams['image.cmap'] = 'rocket'

    checkpoint_path = '/scratch/u95/fl9575/GCE_v2/checkpoints/Fermi_example_add_two_256_BN_bs_256_softplus_pre_gen'

    bin_edges = np.asarray([-np.infty] + np.logspace(-12.5, -7, 21).tolist() + [np.infty])

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    data_folder = os.path.join(checkpoint_path, 'Best_fit_maps_random_GCE')
    pred_out_folder = os.path.join(data_folder, "Predictions")
    pred_out_file = "Pred"

    all_taus = np.linspace(0.05, 0.95, 19)
    n_taus = len(all_taus)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    # Load
    folder_content = os.listdir(data_folder)
    map_files = [m for m in folder_content if ".npy" in m]

    # Load
    NN_pred_mix_all = []
    gce_hist = []
    gce_poiss_ff = []
    for i_m, m in enumerate(map_files):
        NN_pred_data = np.load(os.path.join(pred_out_folder, pred_out_file + "_" + m[:-4]) + ".npz", allow_pickle=True)
        if len(NN_pred_mix_all) == 0:
            NN_pred_mix_all = NN_pred_data["NN_pred_mix_all"]
            gce_hist = NN_pred_data["gce_hist"]
            gce_poiss_ff = NN_pred_data["gce_poiss_ff"]
        else:
            NN_pred_mix_all = np.concatenate([NN_pred_mix_all, NN_pred_data["NN_pred_mix_all"]], axis=1)
            gce_hist = np.concatenate([gce_hist, NN_pred_data["gce_hist"]], axis=0)
            gce_poiss_ff = np.concatenate([gce_poiss_ff, NN_pred_data["gce_poiss_ff"]], axis=0)
    print("Predictions for mixed PS/Poisson data LOADED!")


    class PinballLoss(Loss):
        """Compute the pinball loss"""

        def __init__(self, name="pinball_loss", smoothing=0.0):
            super().__init__(name=name)
            self.smoothing = smoothing
            self.name = name

        def call(self, data, y_pred):
            with tf.name_scope(self.name):
                tau, y_true = data[:, :1], data[:, 1:]

                delta = y_pred - y_true

                # Non-smooth C0 loss (default)
                if self.smoothing == 0.0:
                    mask = tf.cast(tf.greater_equal(delta, tf.zeros_like(delta)), tf.float32) - tau
                    loss = mask * delta

                # Smooth loss
                else:
                    loss = -tau * delta + self.smoothing * tf.math.softplus(delta / self.smoothing)

                final_loss = tf.reduce_mean(loss)

                return final_loss


    # Build NN
    feed_3 = False
    feed_cum = False
    dim_in = 3 * len(bin_centres) if feed_3 else len(bin_centres)
    dim_in_concat = dim_in + 1
    dim_out = 1
    n_layers = 2
    n_hidden = 256
    act_fun = "relu"
    act_fun_final = "sigmoid"
    do_batch_norm = False
    smoothing = 0.001
    batch_size = 2048
    n_epochs = 200
    shuffle = True
    test_frac = 0.2
    tau_mapping = lambda t: (t - 0.5) * 12  # tau |-> tau shown to the NN
    save_path = os.path.join(checkpoint_path, 'Best_fit_maps_random_GCE', 'NN')
    mkdir_p(save_path)

    model = keras.Sequential()
    model.add(layers.Dense(n_hidden, activation=act_fun, input_shape=(dim_in_concat,)))
    if do_batch_norm:
        model.add(layers.BatchNormalization())
    for i_layer in range(n_layers - 1):
        model.add(layers.Dense(n_hidden, activation=act_fun))
        if do_batch_norm:
            model.add(layers.BatchNormalization())
    model.add(layers.Dense(dim_out, activation=act_fun_final))
    model.build()
    model.summary()

    loss_tf = PinballLoss(smoothing=smoothing)

    # Compile NN
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=n_epochs,
                                                                 decay_rate=0.1, staircase=False)

    # Define data and labels
    median_ind = (NN_pred_mix_all.shape[0] - 1) // 2

    x_train_all, x_test_all, y_train, y_test, gce_hist_train, gce_hist_test = \
        train_test_split(NN_pred_mix_all[:, :, :].transpose(1, 0, 2), gce_poiss_ff, gce_hist, test_size=test_frac,)
    x_train_all, x_test_all = x_train_all.transpose(1, 0, 2), x_test_all.transpose(1, 0, 2)
    if feed_3:
        x_train = np.concatenate([x_train_all[0], x_train_all[median_ind], x_train_all[-1]], 1)
        x_test = np.concatenate([x_test_all[0], x_test_all[median_ind], x_test_all[-1]], 1)
    else:
        x_train = x_train_all[median_ind]
        x_test = x_test_all[median_ind]

    tau_test = 0.5 * np.ones((y_test.shape[0], 1))
    tau_test_mapped = tau_mapping(tau_test)
    x_test_w_tau = np.concatenate([tau_test_mapped, x_test], 1)
    y_test_w_tau = np.concatenate([tau_test, y_test], 1)

    # Make changes to training data here, for robustness checks
    # Poiss_flux_dominated_inds = np.argwhere(y_train[:, 0] > 0.5).flatten()
    # PS_flux_dominated_inds = np.argwhere(y_train[:, 0] <= 0.5).flatten()
    # x_train = np.delete(x_train, Poiss_flux_dominated_inds[:len(Poiss_flux_dominated_inds) // 2], axis=0)  # exclude half of the DM dominated maps for testing
    # y_train = np.delete(y_train, Poiss_flux_dominated_inds[:len(Poiss_flux_dominated_inds) // 2], axis=0)  # filename: ..._half_DM_dominated_maps_removed
    # x_train = np.delete(x_train, PS_flux_dominated_inds[:len(PS_flux_dominated_inds) // 2], axis=0)  # exclude half of the PS dominated maps for testing
    # y_train = np.delete(y_train, PS_flux_dominated_inds[:len(PS_flux_dominated_inds) // 2], axis=0)  # filename: ..._half_PS_dominated_maps_removed

    # Define data pipeline
    class DataGenerator(keras.utils.Sequence):
        """Generates data for Keras"""

        def __init__(self, list_IDs, batch_size=16, dim_in=1, dim_out=1, generate_tau=False, shuffle=True):
            """Initialization"""
            self.list_IDs = list_IDs
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.generate_tau = generate_tau
            self.on_epoch_end()

        def __len__(self):
            """Denotes the number of batches per epoch"""
            return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __data_generation(self, list_IDs_temp):
            x = x_train[list_IDs_temp]
            y = y_train[list_IDs_temp]
            if self.generate_tau:
                tau = tf.random.uniform((self.batch_size, 1), 0.0, 1.0)
                if feed_cum:
                    x = tf.cast(tf.cumsum(x, axis=1), dtype=tf.float32)
                x = tf.concat([tau_mapping(tau), x], axis=1)  # scale tau for input
                y = tf.concat([tau, y], axis=1)
            return x, y

        def __getitem__(self, index):
            """Generate one batch of data"""
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data and append tau to input
            x, y = self.__data_generation(list_IDs_temp)

            return x, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

    # Train / load NN
    model_path = os.path.join(save_path, "Model_trained")
    if os.path.exists(model_path):
        # Load and compile
        model.load_weights(os.path.join(model_path, "weights"))
        model.compile(loss=loss_tf, optimizer='adam', metrics=[])
    else:
        # Compile
        model.compile(loss=loss_tf, optimizer='adam', metrics=[])
        mkdir_p(model_path)

        # Train and save
        history = model.fit(x=DataGenerator(np.arange(x_train.shape[0]), batch_size, dim_in, dim_out, generate_tau=True, shuffle=shuffle),
                            epochs=n_epochs, verbose=2)
        model.save_weights(os.path.join(model_path, "weights"))

        # Show training progress
        fig_history, ax_history = plt.subplots(1, 1)
        ax_history.plot(history.history["loss"])
        ax_history.set_xlabel("Epoch")
        ax_history.set_ylabel("Loss")
        fig_history.savefig(os.path.join(save_path, "train_loss.pdf"), bbox_inches="tight")

    # Evaluate for different taus
    n_eval = x_test_all.shape[1]
    eval_preds = np.zeros((n_taus, n_eval))
    # eval_data, eval_label = x_train[median_ind, :n_eval], y_train[:n_eval]
    eval_data, eval_label = x_test_all[median_ind, :n_eval].copy(), y_test[:n_eval].copy()

    for i_tau, tau in enumerate(all_taus):
        if feed_cum:
            eval_data = np.cumsum(eval_data, 1)
        eval_data_w_tau = np.concatenate([tau_mapping(tau) * np.ones((n_eval, 1)), eval_data], 1)
        eval_pred = model.predict(eval_data_w_tau)
        eval_preds[i_tau, :] = eval_pred.squeeze()
        print("tau: {:0.2f}, Coverage: {:0.3f}".format(tau, (eval_pred >= eval_label).mean()))

    # Make a calibration plot
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(4, 4))
    ax_cal.plot([0, 1], [0, 1], "k-", lw=2)
    ax_cal.plot(all_taus, (eval_preds > eval_label.squeeze()).mean(1),
                "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    ax_cal.set_xlim([-0.05, 1.05])
    ax_cal.set_ylim([-0.05, 1.05])
    ax_cal.set_xlabel(r"Confidence level $\alpha$")
    ax_cal.set_ylabel(r"Coverage $p_{\mathrm{cov}}(\alpha)$")
    ax_cal.set_aspect("equal")
    ticks = np.linspace(0, 1, 6)
    ax_cal.set_xticks(ticks)
    ax_cal.set_yticks(ticks)
    plt.tight_layout()
    fig_cal.savefig(os.path.join(save_path, "calibration_plot.pdf"), bbox_inches="tight")

    # Plot some examples (from the dataset it was not calibrated on)
    tau_inds_eval_constraint = [18, 13, 9]  # 0.95, 0.7, 0.5
    plot_start, plot_end = 0, 100
    plot_range = np.arange(plot_start, plot_end)
    width = np.diff(bin_centres).mean()

    plt.ioff()
    for plot_ind in plot_range:
        fig, ax = plt.subplots(1, 1)
        ax.plot(bin_centres, x_test_all[median_ind, plot_ind].cumsum(), "k-", lw=2)  # NN prediction for mixed PS/Poisson map: median

        # Plot cumulative histogram estimates (faint)
        for i_tau in range(len(all_taus)):
            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres)):
                    # Draw the next section of the cumulative histogram in the right colour
                    ax.fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                    y1=x_test_all[i_tau, plot_ind].cumsum()[i],
                                    y2=x_test_all[i_tau + 1, plot_ind].cumsum()[i], color=colors[i_tau], lw=0, alpha=0.5)
        ax.plot(bin_centres, gce_hist_test[plot_ind, :, 0].cumsum(), color="darkslateblue", ls="-", lw=2)  # True label (PS only, 0: GCE)
        ax.axhline(eval_label[plot_ind].squeeze(), color="k", lw=1, ls="--")
        ax.text(-11, 0.05, "True Poisson FF:", ha="left")
        ax.text(-7, 0.05, "{:2.1f}%".format(100 * eval_label[plot_ind].squeeze()), ha="right")
        ax.text(-11, 0.01, "Constraints:", ha="left")
        ax.set_title(str(plot_ind))
        for i_spacing, i_tau in enumerate(tau_inds_eval_constraint):
            ax.text(-7 - (i_spacing * 0.8), 0.01, "{:2.1f}%".format(100 * eval_preds[i_tau, plot_ind]),
                    ha="right")
        print("Poisson fraction: {:2.1f}%".format(100 * eval_label[plot_ind].squeeze()))

    multipage(os.path.join(save_path, "example_constraints.pdf"))
    plt.close("all")
    plt.ion()

    # Try out the method for the Fermi map on some test maps, deriving constraints under the assumption that any tau
    # for the histogram might be the correct one
    max_cum_hist_training = np.max(x_train.cumsum(1), 0)
    q99_cum_hist_training = np.quantile(x_train.cumsum(1), 0.99, 0)
    q975_cum_hist_training = np.quantile(x_train.cumsum(1), 0.975, 0)
    # cmap_fermi = cc.cm.CET_D3_r
    # cmap_fermi = mpl.colors.ListedColormap(sns.color_palette("Spectral", 1000))
    cmap_fermi = cc.cm.CET_R1

    if not feed_3:
        test_range = np.arange(0, 5)
        test_constraints = np.zeros((len(test_range), n_taus, n_taus))  # second index: quantile levels for Fermi prediction
                                                                        # third index: confidence w.r.t. FF constraint
        exclude_hist = q99_cum_hist_training

        for count, i in enumerate(test_range):
            for i_tau_test in range(len(all_taus)):
                test_input = x_test_all[i_tau_test, i, :]
                test_constraint = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)), np.tile(test_input, [n_taus, 1])], 1))
                test_constraints[count, i_tau_test, :] = test_constraint.squeeze()

            fig_test, ax_test = plt.subplots(1, 1, figsize=(6, 6))
            im = ax_test.imshow(100 * test_constraints[count].T, cmap=cmap_fermi, aspect='equal', interpolation='none', vmin=0, vmax=100, origin='lower')
            cbar = fig_test.colorbar(im, fraction=0.0458, pad=0.04)
            cbar.set_label(r"Max. Poisson flux fraction [$\%$]")
            ax_test.set_xlabel(r"Quantile level $\tau$")
            ax_test.set_ylabel("Confidence")
            ticks = [0, 3, 6, 9, 12, 15, 18]
            tick_labels = ["{:#1.2f}".format(t) for t in np.round(all_taus[ticks], 2)]
            ax_test.set_xticks(ticks)
            ax_test.set_xticklabels(tick_labels)
            ax_test.set_yticks(ticks)
            ax_test.set_yticklabels(tick_labels)
            ax_test.set_title("{:2.2f}%".format(100 * y_test[i, 0]))
            # Draw the PDF for each quantile level
            for i_tau_test in range(len(all_taus)):
                ax_test.plot(i_tau_test - x_test_all[i_tau_test, i] / x_test_all[i_tau_test, i].max() + 0.5,
                             n_taus * np.linspace(0, 1, len(bin_centres)) - 0.5, color="k", lw=1)
            cum_hists = x_test_all[:, i, :]
            out_of_prior_bins = np.any(cum_hists > exclude_hist, axis=1)
            if np.any(out_of_prior_bins):
                hatch_ind_min, hatch_ind_max = np.argmax(out_of_prior_bins), n_taus
                rect_exclude = mpl.patches.Rectangle((hatch_ind_min - 0.5, ax_test.get_ylim()[0]),
                                                     hatch_ind_max - hatch_ind_min + 1, np.diff(ax_test.get_ylim()),
                                                     lw=0, fill=True, color="1.0", alpha=1.0)
                ax_test.add_patch(rect_exclude)
            plt.tight_layout()

    # NOW: evaluate on Fermi map
    if not feed_3:
        DRAW_PDFS = False
        exclude_hist = q99_cum_hist_training

        fermi_pred_data = np.load(os.path.join(checkpoint_path, "fermi_prediction.npz"), allow_pickle=True)
        print(fermi_pred_data.files)
        fermi_pred = fermi_pred_data["fermi_pred"][()]
        fermi_hist_gce = fermi_pred["gce_hist"][:, :, 0]
        fermi_input = fermi_hist_gce[median_ind]
        fermi_pred_Poiss_FF = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)),
                                                            np.tile(fermi_input, [n_taus, 1])], 1))

        fermi_constraints = np.zeros((n_taus, n_taus))  # first index: quantile levels for Fermi prediction
                                                        # second index: confidence w.r.t. FF constraint
        for i_tau_fermi in range(len(all_taus)):
            fermi_input = fermi_hist_gce[i_tau_fermi]
            fermi_constraint = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)), np.tile(fermi_input, [n_taus, 1])], 1))
            fermi_constraints[i_tau_fermi, :] = fermi_constraint.squeeze()

        fig_fermi, ax_fermi = plt.subplots(1, 1, figsize=(6, 6))
        im = ax_fermi.imshow(100 * fermi_constraints.T, cmap=cmap_fermi, aspect='equal', interpolation='none', vmin=0, vmax=100, origin='lower')
        ax_fermi.set_xlabel(r"Quantile level $\tau$")
        ax_fermi.set_ylabel(r"Confidence level $\alpha$")
        ticks = [0, 3, 6, 9, 12, 15, 18]
        tick_labels = ["{:#1.2f}".format(t) for t in np.round(all_taus[ticks], 2)]
        ax_fermi.set_xticks(ticks)
        ax_fermi.set_xticklabels(tick_labels)
        ax_fermi.set_yticks(ticks)
        ax_fermi.set_yticklabels(tick_labels)
        if not DRAW_PDFS:
            rect = mpl.patches.Rectangle([median_ind - 0.5, ax_fermi.get_ylim()[0]], 1.04, np.diff(ax_fermi.get_ylim()), fill=False,
                                         ec="k", lw=2)
            ax_fermi.add_patch(rect)

        tol = 0.001
        out_of_prior_bins = np.any(fermi_hist_gce.cumsum(1) > exclude_hist + tol, 1)
        hatch_ind_min, hatch_ind_max = np.argmax(out_of_prior_bins), n_taus
        rect_exclude = mpl.patches.Rectangle((hatch_ind_min - 0.5, ax_fermi.get_ylim()[0]),
                                             hatch_ind_max - hatch_ind_min + 1, np.diff(ax_fermi.get_ylim()),
                                             lw=0, fill=True, color="1.0", alpha=1.0)
        ax_fermi.add_patch(rect_exclude)
        if DRAW_PDFS:
            # Draw the PDF for each quantile level
            for i_tau_fermi in range(len(all_taus)):
                ax_fermi.plot(i_tau_fermi - fermi_hist_gce[i_tau_fermi] / fermi_hist_gce[i_tau_fermi].max() + 0.5,
                              n_taus * np.linspace(0, 1, len(bin_centres)) - 0.5, color="k", lw=1)
            ax_fermi.set_xlim([-0.5, n_taus + 0.5])
        else:
            rect_exclude_h = mpl.patches.Rectangle((hatch_ind_min - 0.5, ax_fermi.get_ylim()[0]),
                                                   hatch_ind_max - hatch_ind_min + 1, np.diff(ax_fermi.get_ylim()), lw=0,
                                                   fill=None, hatch="\\\\\\", color="k", alpha=0.25)
            ax_fermi.add_patch(rect_exclude_h)

        cbar = fig_fermi.colorbar(im, fraction=0.0458, pad=0.04)
        cbar.set_label(r"Max. Poisson flux fraction [$\%$]")
        plt.tight_layout()

        if DRAW_PDFS:
            fig_fermi.savefig(os.path.join(save_path, "Fermi_constraints_w_PDFs.pdf"), bbox_inches="tight")
        else:
            fig_fermi.savefig(os.path.join(save_path, "Fermi_constraints_hatched.pdf"), bbox_inches="tight")

        # Make a line plot of the constraints for the median SCD
        fig_fermi_line, ax_fermi_line = plt.subplots(1, 1, figsize=(6, 2.43))
        ax_fermi_line.plot(all_taus, 100 * fermi_constraints[median_ind, :], lw=0, color="k", marker="x", ms=4)
        ax_fermi_line.set_xticks([np.float32(f) for f in tick_labels])
        ax_fermi_line.set_xticklabels(tick_labels)
        ax_fermi_line.yaxis.set_minor_locator(MultipleLocator(5))
        ax_fermi_line.set_ylabel(r"$\tilde{\eta}_P$ [%]")
        plt.grid(b=True, which='major', axis="y", color='0.5', linestyle='-')
        plt.grid(b=True, which='minor', axis="y", color='0.8', linestyle='-')
        plt.tight_layout()
        fig_fermi_line.savefig(os.path.join(save_path, "Fermi_constraints_median_line.pdf"), bbox_inches="tight")

        # Make plot of the different Fermi quantiles to plot below the Fermi constraint plot
        fig_fermi_pdf = copy.copy(fig_fermi)
        ax_fermi_pdf = fig_fermi_pdf.get_axes()[0]
        ax_fermi_pdf.imshow(np.ones_like(fermi_constraints), cmap="binary", origin='lower')
        max_fac = 0.9
        # Draw the PDF for each quantile level
        for i_tau_fermi in range(len(all_taus)):
            ax_fermi_pdf.fill_betweenx(y=n_taus * np.linspace(0, 1, len(bin_centres)) - 0.5,
                                       x1=i_tau_fermi - max_fac * fermi_hist_gce[i_tau_fermi] / fermi_hist_gce[i_tau_fermi].max() + 0.5,
                                       x2=i_tau_fermi + 0.5, color=colors[i_tau_fermi], lw=0, zorder=5)
            for j_tau_fermi in range(len(all_taus)):
                ax_fermi_pdf.fill_betweenx(y=n_taus * np.linspace(0, 1, len(bin_centres)) - 0.5,
                                           x1=i_tau_fermi - max_fac * fermi_hist_gce[j_tau_fermi] / fermi_hist_gce[j_tau_fermi].max() + 0.5,
                                           x2=i_tau_fermi + 0.5, color=colors[j_tau_fermi], lw=0, zorder=4, alpha=0.1)
        ax_fermi_pdf.set_xlim([-0.5, n_taus - 0.5])
        fig_fermi_pdf.savefig(os.path.join(save_path, "Fermi_constraints_PDFs_only.pdf"), bbox_inches="tight")

        # Find out why constraints are not monotonic at the very low flux end
        # Fainter than the training maps?
        print("Max. cum. histogram of training data [%]: ", np.round(100 * np.max(x_train.cumsum(1), 0), 2))
        print("Cum. histogram of 1st faintest Fermi quantile:", np.round((100 * fermi_hist_gce)[-1].cumsum(), 2))
        print("Cum. histogram of 2nd faintest Fermi quantile:", np.round((100 * fermi_hist_gce)[-2].cumsum(), 2))
        print("Cum. histogram of 3rd faintest Fermi quantile:", np.round((100 * fermi_hist_gce)[-3].cumsum(), 2))

        # -> the faintest 2 are fainter than any of the training histograms in the lowest two flux bins!
        # Does the flux in the first two bins make a difference at all?
        # a)
        fermi_faintest_hist = fermi_hist_gce[-1].copy()
        # b)
        fermi_faintest_hist_no_flux_first_two_bins = fermi_faintest_hist.copy()
        fermi_faintest_hist_no_flux_first_two_bins[:2] = 0
        print("fermi_faintest_hist_no_flux_first_two_bins:",
              np.round((100 * fermi_faintest_hist_no_flux_first_two_bins).cumsum(), 2))

        # Note: now, the histogram doesn't sum up to 1 anymore, but it's just for the sake of testing
        # c)
        fermi_faintest_hist_flux_moved_to_bin_3 = fermi_hist_gce[-1].copy()
        fermi_faintest_hist_flux_moved_to_bin_3[2] += fermi_faintest_hist_flux_moved_to_bin_3[:2].sum()
        fermi_faintest_hist_flux_moved_to_bin_3[:2] = 0
        print("fermi_faintest_hist_flux_moved_to_bin_3:",
              np.round((100 * fermi_faintest_hist_flux_moved_to_bin_3).cumsum(), 2))

        # d)
        hist_only_bin_1 = np.zeros(len(bin_centres))
        hist_only_bin_1[0] = 1

        pred_fermi_faintest = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)),
                                                            np.tile(fermi_faintest_hist, [n_taus, 1])], 1))
        pred_fermi_faintest_no_flux_first_two_bins = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)),
                                                                                   np.tile(fermi_faintest_hist_no_flux_first_two_bins, [n_taus, 1])], 1))
        pred_Fermi_faintest_flux_moved_to_bin_3 = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)),
                                                                                np.tile(fermi_faintest_hist_flux_moved_to_bin_3, [n_taus, 1])], 1))
        pred_hist_only_bin_1 = model.predict(np.concatenate([tau_mapping(np.expand_dims(all_taus, -1)),
                                                                                np.tile(hist_only_bin_1, [n_taus, 1])], 1))
        print(np.round(100 * pred_fermi_faintest, 2))
        print(np.round(100 * pred_fermi_faintest_no_flux_first_two_bins, 2))
        print(np.round(100 * pred_Fermi_faintest_flux_moved_to_bin_3, 2))
        print(np.round(100 * pred_hist_only_bin_1, 2))
        # YES! Higher flux in lowest two bins actually leads to TIGHTER constraints!!!

        # Make a nice plot for four selected simulated test maps
        tau_inds_eval_constraint = [18, 13, 9]  # 0.95, 0.7, 0.5
        plot_inds = [61, 94, 9, 80]
        width = np.diff(bin_centres).mean()

        fig, axs = plt.subplots(2, 2, figsize=(5.5, 5.2), sharex="none", sharey="none")
        axs = axs.flatten()
        for i_ax, plot_ind in enumerate(plot_inds):
            ax = axs[i_ax]
            ax.plot(bin_centres, x_test_all[median_ind, plot_ind].cumsum(), lw=0, marker="v", ms=4, mec="k", mfc="white")  # NN prediction for mixed PS/Poisson map: median

            # Plot cumulative histogram estimates (faint)
            for i_tau in range(len(all_taus)):
                if i_tau < n_taus - 1:
                    # Draw the next section of the cumulative histogram in the right colour
                    for i in range(len(bin_centres)):
                        # Draw the next section of the cumulative histogram in the right colour
                        ax.fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                        y1=x_test_all[i_tau, plot_ind].cumsum()[i],
                                        y2=x_test_all[i_tau + 1, plot_ind].cumsum()[i], color=colors[i_tau], lw=0,
                                        alpha=0.5)
            ax.plot(bin_centres, gce_hist_test[plot_ind, :, 0].cumsum(), color="darkslateblue", ls="none",
                    marker="o", ms=4)  # True label (PS only, 0: GCE)
            ax.axhline(eval_label[plot_ind].squeeze(), color="0.5", lw=1, ls="--")
            ax.text(-6.5, 0.8, "{:2.1f}%".format(100 * eval_label[plot_ind].squeeze()),
                    ha="right", size="small", color="0.5")

            for i_spacing, i_tau in enumerate(tau_inds_eval_constraint):
                ax.text(-6.5, 0.01 + i_spacing * 0.1, "{:2.1f}%".format(100 * eval_preds[i_tau, plot_ind]), ha="right",
                        size="small")
            if i_ax % 2 > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel("CDF")
            if i_ax < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel(r"$\log_{10} \ F$")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(os.path.join(save_path, "selected_constraints.pdf"), bbox_inches="tight")

        # Hard-code mean exposure here (will be needed below)
        mean_exp = 89266586245.44131  # HARD-CODED HERE

        # NOW: What happens to injection example?
        inj_data_PS = np.load(os.path.join(checkpoint_path, "Fermi_injection_data_PS", "Fermi_injection_data_PS_all.npz"), allow_pickle=True)
        f_vec = np.logspace(-1, 1, 5) / mean_exp  # flux per PS array
        inj_data_DM = np.load(os.path.join(checkpoint_path, "Fermi_injection_data.npz"), allow_pickle=True)

        all_FFs_inj_PS = inj_data_PS["all_FFs_inj"][()]
        all_FF_stds_inj_PS = inj_data_PS["all_FF_stds_inj"][()]
        all_hists_inj_PS = inj_data_PS["all_hists_inj"][()]  # shape: n_xis x n_taus x n_maps x n_bins x 2
        xi_vec_PS = inj_data_PS["xi_vec"][()]

        all_FFs_inj_DM = inj_data_DM["all_FFs_inj"][()]
        all_FF_stds_inj_DM = inj_data_DM["all_FF_stds_inj"][()]
        all_hists_inj_DM = inj_data_DM["all_hists_inj"][()]  # shape: n_xis x n_taus x n_maps x n_bins x 2
        xi_vec_DM = inj_data_DM["xi_vec"][()]

        assert np.all(xi_vec_PS == xi_vec_DM), "xi_vecs are different!"
        xi_vec = xi_vec_PS

        # Predict PS injection
        inj_pred_median_PS_all = []
        for i_f in range(len(f_vec)):
            hists_inj_gce_PS = all_hists_inj_PS[:, i_f, :, :, :, 0]
            hists_inj_gce_median_PS = np.median(hists_inj_gce_PS, 2)  # median over the maps
            hists_inj_gce_flat_PS = hists_inj_gce_PS.reshape([np.product(hists_inj_gce_PS.shape[:3]), -1])
            inj_preds_flat_PS = np.zeros((hists_inj_gce_flat_PS.shape[0], n_taus))
            n_preds_inj_PS = inj_preds_flat_PS.shape[0]

            for i_tau, tau in enumerate(all_taus):
                pred_inj_PS = model.predict(np.concatenate([np.tile(tau_mapping(tau), [n_preds_inj_PS, 1]), hists_inj_gce_flat_PS], 1))
                inj_preds_flat_PS[:, i_tau] = pred_inj_PS.squeeze()

            inj_preds_PS = inj_preds_flat_PS.reshape(list(hists_inj_gce_PS.shape[:3]) + [n_taus])  # n_xis x n_taus x n_maps x n_confidence
            inj_preds_median_PS = np.median(inj_preds_PS, 2)  # median over the maps
            inj_pred_median_PS_all.append(inj_preds_median_PS)

            print("Inj. FF: 0.00", "   50%: {:2.1f}%".format(100 * fermi_constraints[median_ind, 9]),
                  "70%: {:2.1f}%".format(100 * fermi_constraints[median_ind, 13]),
                  "95%: {:2.1f}%".format(100 * fermi_constraints[median_ind, 18]))
            for i_xi, xi in enumerate(xi_vec):
                print("Inj. FF:", xi, "   50%: {:2.1f}%".format(100 * inj_preds_median_PS[i_xi, median_ind, 9]),
                      "70%: {:2.1f}%".format(100 * inj_preds_median_PS[i_xi, median_ind, 13]),
                      "95%: {:2.1f}%".format(100 * inj_preds_median_PS[i_xi, median_ind, 18]))

        inj_pred_median_PS_all = np.asarray(inj_pred_median_PS_all).transpose([1, 0, 2, 3])  # n_xis x n_f x n_tau_hist x n_alpha

        # Predict DM injection
        hists_inj_gce_DM = all_hists_inj_DM[:, :, :, :, 0]
        hists_inj_gce_median_DM = np.median(hists_inj_gce_DM, 2)  # median over the maps
        hists_inj_gce_flat_DM = hists_inj_gce_DM.reshape([np.product(hists_inj_gce_DM.shape[:3]), -1])
        inj_preds_flat_DM = np.zeros((hists_inj_gce_flat_DM.shape[0], n_taus))
        n_preds_inj_DM = inj_preds_flat_DM.shape[0]

        for i_tau, tau in enumerate(all_taus):
            pred_inj_DM = model.predict(
                np.concatenate([np.tile(tau_mapping(tau), [n_preds_inj_DM, 1]), hists_inj_gce_flat_DM], 1))
            inj_preds_flat_DM[:, i_tau] = pred_inj_DM.squeeze()

        inj_preds_DM = inj_preds_flat_DM.reshape(
            list(hists_inj_gce_DM.shape[:3]) + [n_taus])  # n_xis x n_taus x n_maps x n_confidence
        inj_preds_median_DM = np.median(inj_preds_DM, 2)  # median over the maps

        print("Inj. FF: 0.00", "   50%: {:2.1f}%".format(100 * fermi_constraints[median_ind, 9]),
              "70%: {:2.1f}%".format(100 * fermi_constraints[median_ind, 13]),
              "95%: {:2.1f}%".format(100 * fermi_constraints[median_ind, 18]))
        for i_xi, xi in enumerate(xi_vec):
            print("Inj. FF:", xi, "   50%: {:2.1f}%".format(100 * inj_preds_median_DM[i_xi, median_ind, 9]),
                  "70%: {:2.1f}%".format(100 * inj_preds_median_DM[i_xi, median_ind, 13]),
                  "95%: {:2.1f}%".format(100 * inj_preds_median_DM[i_xi, median_ind, 18]))

        # Combined PS / DM injection plot FOR PAPER!
        fig_inj_1d, ax_inj_1d = plt.subplots(1, 1, figsize=(6, 4.5))
        inj_preds_median_median_DM = inj_preds_median_DM[:, median_ind, :]
        inj_preds_median_median_PS_all = inj_pred_median_PS_all[:, :, median_ind, :]
        colors_inj = cc.cm.fire(np.linspace(0, 1, len(f_vec) + 3))[1:-1]
        conf_to_plot = -1  # 95% confidence (index -1 / 18)
        xi_vec_with_0 = np.asarray([0] + list(xi_vec))
        this_constraint = [fermi_constraints[median_ind, conf_to_plot]] + list(inj_preds_median_median_DM[:, conf_to_plot])
        ax_inj_1d.plot(100 * xi_vec_with_0, 100 * np.asarray(this_constraint), "k", lw=1.5)
        for i_f in range(len(f_vec)):
            this_constraint = [fermi_constraints[median_ind, conf_to_plot]] + list(inj_preds_median_median_PS_all[:, i_f, conf_to_plot])
            ax_inj_1d.plot(100 * xi_vec_with_0, 100 * np.asarray(this_constraint), color=colors_inj[i_f], lw=1.5)
        ax_inj_1d.set_ylim([32.5, 102.5])
        ax_inj_1d.set_xlabel("Injected flux fraction [%]")
        ax_inj_1d.set_ylabel(r"$\tilde{\eta}_P(\mathbf{x}; 0.95)$")
        ax_inj_1d.legend(["Poisson", "0.10", "0.32", "1.0", "3.2", "10"], fontsize="small")
        plt.tight_layout()
        fig_inj_1d.savefig(os.path.join(save_path, "inj_plot_constraints_DM_and_PS_zoomed.pdf"), bbox_inches="tight")

        # For DM: Produce the same plot as above, for the median histograms (over the maps)
        exclude_hist = q99_cum_hist_training
        for count in np.arange(len(xi_vec)):
            fig_inj, ax_inj = plt.subplots(1, 1, figsize=(6, 6))
            im = ax_inj.imshow(100 * inj_preds_median_DM[count].T, cmap=cc.cm.CET_D3_r, aspect='equal',
                               interpolation='none', vmin=0, vmax=100, origin='lower')
            cbar = fig_inj.colorbar(im, fraction=0.0458, pad=0.04)
            cbar.set_label(r"Max. Poisson flux fraction [$\%$]")
            ax_inj.set_xlabel(r"Quantile level $\tau$")
            ax_inj.set_ylabel("Confidence")
            ticks = [0, 3, 6, 9, 12, 15, 18]
            tick_labels = ["{:#1.2f}".format(t) for t in np.round(all_taus[ticks], 2)]
            ax_inj.set_xticks(ticks)
            ax_inj.set_xticklabels(tick_labels)
            ax_inj.set_yticks(ticks)
            ax_inj.set_yticklabels(tick_labels)
            # Draw the PDF for each quantile level
            for i_tau_test in range(len(all_taus)):
                ax_inj.plot(i_tau_test - hists_inj_gce_median_DM[count, i_tau_test] / x_test_all[count, i_tau_test].max() + 0.5,
                             n_taus * np.linspace(0, 1, len(bin_centres)) - 0.5, color="k", lw=1)
            cum_hists = hists_inj_gce_median_DM[count]
            out_of_prior_bins = np.any(cum_hists > exclude_hist, axis=1)
            if np.any(out_of_prior_bins):
                hatch_ind_min, hatch_ind_max = np.argmax(out_of_prior_bins), n_taus
                rect_exclude = mpl.patches.Rectangle((hatch_ind_min - 0.5, ax_inj.get_ylim()[0]),
                                                     hatch_ind_max - hatch_ind_min + 1, np.diff(ax_inj.get_ylim()),
                                                     lw=0, fill=True, color="1.0", alpha=1.0)
                ax_inj.add_patch(rect_exclude)
            plt.tight_layout()

        # Make a plot as a function of the injected DM / PS flux, for different confidence levels, for the median prediction
        fig_inj_1d, ax_inj_1d = plt.subplots(1, 1, figsize=(6, 6))
        conf_to_plot = [9, 13, 18]  # np.arange(0, 21, 3) / np.arange(18)
        lw_1d = 1.5
        lss = ["-", "--", (0, (1, 1))][::-1]
        alphas = [1.0, 0.8, 0.6][::-1]

        # First: DM
        for i_conf, conf in enumerate(conf_to_plot):
            this_constraint = [fermi_constraints[median_ind, conf]] + list(inj_preds_median_median_DM[:, conf])
            ax_inj_1d.plot(xi_vec_with_0, this_constraint, "k-", alpha=alphas[i_conf], lw=lw_1d, ls=lss[i_conf])
            # Plot a circle at Fermi value
            ax_inj_1d.plot(xi_vec_with_0[0], fermi_constraints[median_ind, conf], lw=0, marker="o", ms=8, mec="k", mfc="white", zorder=3)

        # Now: PS
        which = "PS"
        xi_vec_with_0 = np.asarray([0] + list(xi_vec))
        for i_f in range(len(f_vec)):
            for i_conf, conf in enumerate(conf_to_plot):
                if i_conf == 0:
                    colors_inj = cc.cm.kgy(np.linspace(0, 1, len(f_vec) + 3))[1:-1]
                elif i_conf == 1:
                    colors_inj = cc.cm.kbc(np.linspace(0, 1, len(f_vec) + 3))[1:-1]
                elif i_conf == 2:
                    colors_inj = cc.cm.fire(np.linspace(0, 1, len(f_vec) + 3))[1:-1]

                this_constraint = [fermi_constraints[median_ind, conf]] + list(inj_preds_median_median_PS_all[:, i_f, conf])
                ax_inj_1d.plot(xi_vec_with_0, this_constraint, "-", color=colors_inj[i_f], alpha=alphas[i_conf], lw=lw_1d, ls=lss[i_conf])

        # Now: evaluate on best-fit maps.
        # Load best-fit predictions
        pred_files = ["best_fit_pred", "best_fit_smooth_GCE_pred"]
        try:
            best_fit_pred = np.load(os.path.join(TEST_EXP_PATH, pred_files[0] + ".npy"), allow_pickle=True)[()]
            best_fit_pred_smooth = np.load(os.path.join(TEST_EXP_PATH, pred_files[1] + ".npy"), allow_pickle=True)[()]
        except FileNotFoundError:
            raise FileNotFoundError("Run the script 'save_best_fit_prediction.py' first!")

        # Now: GCE histograms
        hist_best_fit = best_fit_pred["gce_hist"]
        hist_best_fit_smooth = best_fit_pred_smooth["gce_hist"]
        hist_best_fit_median_gce = hist_best_fit[median_ind, :, :, 0]
        hist_best_fit_median_gce_smooth = hist_best_fit_smooth[median_ind, :, :, 0]
        n_best_fit_maps = hist_best_fit.shape[1]

        median_hist_median_cum = np.quantile(hist_best_fit[median_ind, :, :, 0].cumsum(1), 0.5, 0)
        median_hist_16_cum = np.quantile(hist_best_fit[median_ind, :, :, 0].cumsum(1), 0.16, 0)
        median_hist_84_cum = np.quantile(hist_best_fit[median_ind, :, :, 0].cumsum(1), 0.84, 0)

        median_hist_median_cum_smooth = np.quantile(hist_best_fit_smooth[median_ind, :, :, 0].cumsum(1), 0.5, 0)
        median_hist_16_cum_smooth = np.quantile(hist_best_fit_smooth[median_ind, :, :, 0].cumsum(1), 0.16, 0)
        median_hist_84_cum_smooth = np.quantile(hist_best_fit_smooth[median_ind, :, :, 0].cumsum(1), 0.84, 0)

        # Make a plot
        fig_real_vs_bestfit_cdf, ax_real_vs_bestfit_cdf = plt.subplots(1, 1, figsize=(4.85, 3.81))
        # Plot Fermi CDF prediction (median)
        ax_real_vs_bestfit_cdf.plot(bin_centres, fermi_hist_gce[median_ind].cumsum(), "k", lw=0, ms=4, marker="x", zorder=3)
        # Plot simulated best-fit data
        width = np.diff(bin_centres)[0]
        ax_real_vs_bestfit_cdf.hlines(median_hist_median_cum, bin_centres - width / 2, bin_centres + width / 2,
                                      lw=1.5, color="k", zorder=2)
        ax_real_vs_bestfit_cdf.fill_between(x=bin_centres - width / 2, y1=median_hist_16_cum, y2=median_hist_84_cum,
                                            ec="k", step="post", alpha=0.3, lw=1.5, fc="#ffd31e")
        # Plot simulated best-fit data with smooth GCE
        ax_real_vs_bestfit_cdf.hlines(median_hist_median_cum_smooth, bin_centres - width / 2, bin_centres + width / 2,
                                      lw=1.5, color="k", zorder=2, ls="--")
        ax_real_vs_bestfit_cdf.fill_between(x=bin_centres - width / 2, y1=median_hist_16_cum_smooth, y2=median_hist_84_cum_smooth,
                                            ec="k", step="post", alpha=0.3, lw=1.5, fc="#ff6600")
        ax_real_vs_bestfit_cdf.set_xlim([-13, -9])
        # 1ph line
        one_ph_flux = np.log10(1 / mean_exp)
        ax_real_vs_bestfit_cdf.axvline(one_ph_flux, color="orange", ls="--", zorder=4)
        # 3FGL box
        rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                     linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
        ax_real_vs_bestfit_cdf.add_patch(rect)

        # Build twin axes and set limits
        def F2S(x):
            return 10.0 ** x * mean_exp
        twin_axes = ax_real_vs_bestfit_cdf.twiny()
        twin_axes.plot(F2S(bin_centres), fermi_hist_gce[median_ind].cumsum(), color="none", lw=0)

        twin_axes.set_xlim(F2S(np.asarray([-13, -9])))
        twin_axes.set_xlabel(r"$\bar{S}$")
        twin_axes.set_xscale("log")
        twin_axes.set_ylim([-0.075, 1.075])
        ax_real_vs_bestfit_cdf.set_xlabel(r"$\log_{10} \ F$")
        ax_real_vs_bestfit_cdf.set_ylabel("CDF")
        plt.tight_layout()
        fig_real_vs_bestfit_cdf.savefig(os.path.join(save_path, "cdf_fermi_compared_with_best_fit.pdf"), bbox_inches="tight")

        # Get constraints
        best_fit_constraints = np.zeros((n_taus, n_best_fit_maps))
        best_fit_constraints_smooth = np.zeros((n_taus, n_best_fit_maps))
        for i_tau in range(len(all_taus)):
            best_fit_constraints[i_tau, :] = model.predict(np.concatenate([np.tile(tau_mapping(all_taus[i_tau]),
                                                                                   [n_best_fit_maps, 1]), hist_best_fit_median_gce], 1)).squeeze()
            best_fit_constraints_smooth[i_tau, :] = model.predict(np.concatenate([np.tile(tau_mapping(all_taus[i_tau]),
                                                                                          [n_best_fit_maps, 1]), hist_best_fit_median_gce_smooth], 1)).squeeze()

        # Get quantiles of constraints over the maps
        best_fit_constraints_16 = np.quantile(best_fit_constraints, 0.16, axis=1)
        best_fit_constraints_50 = np.quantile(best_fit_constraints, 0.5, axis=1)
        best_fit_constraints_84 = np.quantile(best_fit_constraints, 0.84, axis=1)
        best_fit_constraints_16_smooth = np.quantile(best_fit_constraints_smooth, 0.16, axis=1)
        best_fit_constraints_50_smooth = np.quantile(best_fit_constraints_smooth, 0.5, axis=1)
        best_fit_constraints_84_smooth = np.quantile(best_fit_constraints_smooth, 0.84, axis=1)

        # Make a plot of the constraints for the real Fermi data, best-fit, and best-fit with smooth GCE
        fig_real_vs_bestfit_constraint, ax_real_vs_bestfit_constraint = plt.subplots(1, 1, figsize=(4.85, 3.81))
        ax_real_vs_bestfit_constraint.plot(all_taus, 100 * fermi_constraints[median_ind, :], lw=0, color="k", marker="x", ms=4, zorder=3)  # real data
        ax_real_vs_bestfit_constraint.plot(all_taus, 100 * best_fit_constraints_50, lw=1.5, color="k", zorder=2)
        ax_real_vs_bestfit_constraint.fill_between(x=all_taus, y1=100 * best_fit_constraints_16, y2=100 * best_fit_constraints_84,
                                                   color="k", alpha=0.3)  # simulated best-fit data
        ax_real_vs_bestfit_constraint.plot(all_taus, 100 * best_fit_constraints_50_smooth, lw=1.5, color="k", zorder=2, ls="--")
        ax_real_vs_bestfit_constraint.fill_between(x=all_taus, y1=100 * best_fit_constraints_16_smooth, y2=100 * best_fit_constraints_84_smooth,
                                                   color="k", alpha=0.3)  # simulated best-fit data with smooth GCE
        ticks = [0, 3, 6, 9, 12, 15, 18]
        tick_labels = ["{:#1.2f}".format(t) for t in np.round(all_taus[ticks], 2)]
        ax_real_vs_bestfit_constraint.set_xticks([np.float32(f) for f in tick_labels])
        ax_real_vs_bestfit_constraint.set_xticklabels(tick_labels)
        ax_real_vs_bestfit_constraint.yaxis.set_minor_locator(MultipleLocator(5))
        ax_real_vs_bestfit_constraint.set_ylabel(r"$\tilde{\eta}_P$ [%]")
        ax_real_vs_bestfit_constraint.set_xlabel(r"$\alpha$")
        plt.grid(b=True, which='major', axis="y", color='0.5', linestyle='-')
        plt.grid(b=True, which='minor', axis="y", color='0.9', linestyle='-')
        plt.tight_layout()
        fig_real_vs_bestfit_constraint.savefig(os.path.join(save_path, "constraints_fermi_compared_with_best_fit.pdf"), bbox_inches="tight")

        # Also: constraints for mismodelling data
        data_out_file_mis = "evaluate_NN_mismodelling_data"
        data_mis = np.load(os.path.join(TEST_EXP_PATH, data_out_file_mis + ".npy"), allow_pickle=True)[()]
        pred_hist_all_mis = data_mis["gce_hist"]  # n_mismodelling_cases x n_taus x n_realisations x n_bins x 2 (GCE, disk)
        pred_FF_all_mis = data_mis["logits_mean"]  # n_mismodelling_cases x n_taus x n_realisations x n_templates
        pred_covar_all_mis = data_mis["covar"]  # n_mismodelling_cases x n_taus x n_realisations x n_templates x n_templates
        n_mis = pred_hist_all_mis.shape[0]
        n_mis_maps = pred_hist_all_mis.shape[2]

        pred_hist_median_mis_GCE = pred_hist_all_mis[:, median_ind, :, :, 0]
        mis_constraints_for_median_SCD = np.zeros((n_mis, n_taus, n_mis_maps))
        for i_mis in range(n_mis):
            for i_tau in range(len(all_taus)):
                mis_constraints_for_median_SCD[i_mis, i_tau, :] = model.predict(np.concatenate([np.tile(tau_mapping(all_taus[i_tau]), [n_mis_maps, 1]), pred_hist_median_mis_GCE[i_mis, :, :]], 1)).squeeze()

        mis_constraints_for_median_SCD_median = np.median(mis_constraints_for_median_SCD, axis=2)
        fig_mis, ax_mis = plt.subplots(1, 1)
        for i_mis in range(n_mis):
            ax_mis.plot(all_taus, mis_constraints_for_median_SCD_median[i_mis])

        leg_str = ["Default", "Thick disk", r"Bubbles$^*$", "Model A", "Model F", "p6v11", r"GCE $\gamma = 1.0$"]
        ax_mis.legend(leg_str)
