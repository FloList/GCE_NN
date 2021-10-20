"""
This script computes constraints on the Poissonian flux fraction using the SIMPLE (less powerful) estimator
discussed in arXiv:2107.09070.
Difference w.r.t. constrain_Poisson_flux_iso_with_NN_py:
ALL the quantiles are provided as an input!
"""
import shutil
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.losses import Loss
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import os
import copy
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'

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

    NO_PSF = True  # deactivate PSF?
    DO_FAINT = True  # take the NN trained on fainter maps (NO_PSF = True only)?

    if NO_PSF:
        if DO_FAINT:
            TEST_EXP_PATH = "/scratch/u95/fl9575/GCE_v2/checkpoints/" \
                            "Iso_maps_combined_add_two_faint_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
        else:
            TEST_EXP_PATH = "/scratch/u95/fl9575/GCE_v2/checkpoints/" \
                            "Iso_maps_combined_add_two_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    else:
        if DO_FAINT:
            raise NotImplementedError
        TEST_EXP_PATH = "/scratch/u95/fl9575/GCE_v2/checkpoints/Iso_maps_combined_add_two_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)


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

    checkpoint_path = TEST_EXP_PATH
    if DO_FAINT:
        bin_edges = np.asarray([-np.infty] + list(np.logspace(-2.5, 2, 25)) + [np.infty])
    else:
        bin_edges = np.asarray([-np.infty] + list(np.logspace(-1.5, 2, 21)) + [np.infty])

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    data_folder = os.path.join(checkpoint_path, 'Mixed_PS_Poisson')
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
    feed_cum = False
    dim_in = n_taus * len(bin_centres)
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
    save_path = os.path.join(checkpoint_path, 'Mixed_PS_Poisson', 'NN')
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
    x_train = np.reshape(x_train_all, [x_train_all.shape[0], np.product(x_train_all.shape[1:])])
    x_test = np.reshape(x_test_all, [x_test_all.shape[0], np.product(x_test_all.shape[1:])])

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
    model_path = os.path.join(save_path, "Model_trained_marginalised")
    if os.path.exists(model_path):
        # Load and compile
        model.load_weights(os.path.join(model_path, "weights"))
        model.compile(loss=loss_tf, optimizer='adam', metrics=[])
    else:
        # Compile
        model.compile(loss=loss_tf, optimizer='adam', metrics=[])
        mkdir_p(model_path)

        # Train and save
        history = model.fit(x=DataGenerator(np.arange(x_train.shape[0]), batch_size, dim_in, dim_out,
                                            generate_tau=True, shuffle=shuffle), epochs=n_epochs, verbose=2)
        model.save_weights(os.path.join(model_path, "weights"))

        # Show training progress
        fig_history, ax_history = plt.subplots(1, 1)
        ax_history.plot(history.history["loss"])
        ax_history.set_xlabel("Epoch")
        ax_history.set_ylabel("Loss")
        fig_history.savefig(os.path.join(save_path, "train_loss_marginalised.pdf"), bbox_inches="tight")

    # Evaluate for different taus
    n_eval = x_test_all.shape[0]
    eval_preds = np.zeros((n_taus, n_eval))
    # eval_data, eval_label = x_train[median_ind, :n_eval], y_train[:n_eval]
    eval_data, eval_label = np.reshape(x_test_all[:n_eval], [n_eval, -1]).copy(), y_test[:n_eval].copy()

    for i_tau, tau in enumerate(all_taus):
        if feed_cum:
            eval_data = np.cumsum(eval_data, 1)
        eval_data_w_tau = np.concatenate([tau_mapping(tau) * np.ones((n_eval, 1)), eval_data], 1)
        eval_pred = model.predict(eval_data_w_tau)
        eval_preds[i_tau, :] = eval_pred.squeeze()
        print("tau: {:0.2f}, Coverage: {:0.3f}".format(tau, (eval_pred >= eval_label).mean()))

    # Make a calibration plot
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(4, 4))
    ax_cal.plot(all_taus, all_taus, "k-", lw=2)
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
    fig_cal.savefig(os.path.join(save_path, "calibration_plot_marginalised.pdf"), bbox_inches="tight")

    # Plot some examples (from the dataset it was not calibrated on)
    tau_inds_eval_constraint = [18, 13, 9]  # 0.95, 0.7, 0.5
    plot_start, plot_end = 0, 100
    plot_range = np.arange(plot_start, plot_end)
    width = np.diff(bin_centres).mean()

    plt.ioff()
    for plot_ind in plot_range:
        fig, ax = plt.subplots(1, 1)
        ax.plot(bin_centres, x_test_all[plot_ind, median_ind].cumsum(), "k-", lw=2)  # NN prediction for mixed PS/Poisson map: median

        # Plot cumulative histogram estimates (faint)
        for i_tau in range(len(all_taus)):
            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres)):
                    # Draw the next section of the cumulative histogram in the right colour
                    ax.fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                    y1=x_test_all[plot_ind, i_tau].cumsum()[i],
                                    y2=x_test_all[plot_ind, i_tau + 1].cumsum()[i],
                                    color=colors[i_tau], lw=0, alpha=0.5)
        ax.plot(bin_centres, gce_hist_test[plot_ind, :, 0].cumsum(), color="darkslateblue", ls="-", lw=2)  # True label (PS only, 0: GCE)
        ax.axhline(eval_label[plot_ind].squeeze(), color="k", lw=1, ls="--")
        ax.text(0, 0.05, "True Poisson FF:", ha="left")
        ax.text(2.3, 0.05, "{:2.1f}%".format(100 * eval_label[plot_ind].squeeze()), ha="right")
        ax.text(0, 0.01, "Constraints:", ha="left")
        ax.set_title(str(plot_ind))
        for i_spacing, i_tau in enumerate(tau_inds_eval_constraint):
            ax.text(2.3 - (i_spacing * 0.5), 0.01, "{:2.1f}%".format(100 * eval_preds[i_tau, plot_ind]),
                    ha="right")
        print("Poisson fraction: {:2.1f}%".format(100 * eval_label[plot_ind].squeeze()))

    multipage(os.path.join(save_path, "example_constraints_marginalised.pdf"))
    plt.close("all")
    plt.ion()
