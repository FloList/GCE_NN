"""
Systematically evaluate constraints on the Poissonian flux fraction as a function of the PS brightness.
Part 2:
Evaluate trained NN h^nu, which takes the predicted histograms as inputs and predicts the Poissonian FFs.
"""
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    DO_FAINT_EVALUATE_ON_BRIGHT = False  # if True: take the NN trained on faint SCDs, but still use the brighter priors for constraining the Poisson flux

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
    data_folder_orig = copy.copy(data_folder)

    # Use data folder of the BRIGHT priors?
    if DO_FAINT and DO_FAINT_EVALUATE_ON_BRIGHT:
        data_folder = data_folder.replace("_faint", "")

    if DO_FAINT and DO_FAINT_EVALUATE_ON_BRIGHT:
        pred_out_folder = os.path.join(data_folder_orig, "Predictions_bright_data")
    else:
        pred_out_folder = os.path.join(data_folder_orig, "Predictions")

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
    DO_MARGINALISED = False
    if DO_MARGINALISED:
        dim_in = n_taus * len(bin_centres)
    else:
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
    save_path = os.path.join(checkpoint_path, 'Mixed_PS_Poisson', 'NN')
    if DO_FAINT_EVALUATE_ON_BRIGHT:
        save_path += "_bright_data"
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
            return None

        def __getitem__(self, index):
            return None

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

    # Load NN
    model_name = "Model_trained_marginalised" if DO_MARGINALISED else "Model_trained"
    model_path = os.path.join(save_path, model_name)
    if os.path.exists(model_path):
        # Load and compile
        model.load_weights(os.path.join(model_path, "weights"))
        model.compile(loss=loss_tf, optimizer='adam', metrics=[])
    else:
        raise FileNotFoundError("No NN weights found! Run 'constrain_Poisson_flux_iso_with_NN.py' first!")

    # Load the delta dN/dlogF histograms
    sys_data_path = os.path.join(checkpoint_path, "Mixed_PS_Poisson", "Systematic")
    sys_data_pred_path = os.path.join(sys_data_path, "Predictions")
    sys_data_pred_file = os.path.join(sys_data_pred_path, "Pred.npy")

    try:
        sys_data = np.load(sys_data_pred_file)
    except FileNotFoundError:
        raise FileNotFoundError("Run 'systematically_evaluate_constraints_part_1.py' first!")

    # Get median prediction and flatten
    _, n_Poiss_frac, n_counts_per_PS, n_realisations, _ = sys_data.shape
    sys_data_median = sys_data[median_ind, :, :, :, :]  # Median tau
    sys_data_median_flat = np.reshape(sys_data_median, [-1, len(bin_centres)])
    sys_data_flat = sys_data.transpose([1, 2, 3, 0, 4]).reshape([-1, n_taus * len(bin_centres)])

    # Evaluate for different confidence levels alpha (here: tau, somewhat confusing)
    n_eval_sys = sys_data_median_flat.shape[0]
    sys_preds_flat = np.zeros((n_taus, n_eval_sys))

    if DO_MARGINALISED:
        for i_tau, tau in enumerate(all_taus):
            sys_data_loc = sys_data_flat
            sys_data_w_tau = np.concatenate([tau_mapping(tau) * np.ones((n_eval_sys, 1)), sys_data_loc], 1)
            sys_pred = model.predict(sys_data_w_tau)
            sys_preds_flat[i_tau, :] = sys_pred.squeeze()
    else:
        for i_tau, tau in enumerate(all_taus):
            if feed_cum:
                sys_data_loc = np.cumsum(sys_data_median_flat, 1)
            else:
                sys_data_loc = sys_data_median_flat
            sys_data_w_tau = np.concatenate([tau_mapping(tau) * np.ones((n_eval_sys, 1)), sys_data_loc], 1)
            sys_pred = model.predict(sys_data_w_tau)
            sys_preds_flat[i_tau, :] = sys_pred.squeeze()

    # Bring to shape n_taus x n_Poiss_frac x n_counts_per_PS x n_realisations
    sys_preds = np.transpose(np.reshape(sys_preds_flat.T, [n_Poiss_frac, n_counts_per_PS, n_realisations, n_taus]), [3, 0, 1, 2])

    # Define counts per PS and Poisson FF arrays
    counts_per_PS_ary = np.logspace(-1, 3, 11)[:7]
    Poiss_fracs = np.linspace(0.0, 1.0, 6)

    # Define the colour map
    x_vec = np.asarray([0, 0.5, 1])
    cmap_orig = copy.copy(cc.cm.CET_D3_r)
    cmap_3_vals = cmap_orig(x_vec)

    cmap_gkr_orig = copy.copy(cc.cm.diverging_gkr_60_10_c40_r)
    cmap_gkr_3_vals = cmap_gkr_orig(x_vec)
    cmap_new_3_vals = np.vstack([cmap_3_vals[0], cmap_gkr_3_vals[1], cmap_3_vals[2]])
    N_interp = 256
    cmap_new_ary = np.asarray([np.interp(np.linspace(0, 1, N_interp), x_vec, cmap_new_3_vals[:, i]) for i in range(4)]).T
    cmap_new = mpl.colors.ListedColormap(cmap_new_ary)

    #colors_constraint = cc.cm.diverging_gkr_60_10_c40_r(np.linspace(0, 1, n_Poiss_frac))
    colors_constraint = cmap_new(np.linspace(0, 1, n_Poiss_frac))
    counts_per_PS_ary_max_ind = 7

    # Make a plot
    if DO_FAINT and DO_FAINT_EVALUATE_ON_BRIGHT:
        out_file_plot = "systematic_constraints_NN_small_bright_data.pdf"
    else:
        out_file_plot = "systematic_constraints_NN_small.pdf"
    plot_ind_tau = 18  # -1: 95% confidence
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.2))
    ax.set_xscale("log")
    x_values = counts_per_PS_ary[:counts_per_PS_ary_max_ind]
    lw = 1.5
    for i_Poiss_frac, Poiss_frac in enumerate(Poiss_fracs):
        median_constraint = np.median(sys_preds[plot_ind_tau, i_Poiss_frac, :counts_per_PS_ary_max_ind, :], 1)  # median over the maps
        scatter_low = median_constraint \
                      - np.quantile(sys_preds[plot_ind_tau, i_Poiss_frac, :counts_per_PS_ary_max_ind, :], 0.16, 1)
        scatter_high = np.quantile(sys_preds[plot_ind_tau, i_Poiss_frac, :counts_per_PS_ary_max_ind, :], 0.84, 1) \
                       - median_constraint
        yerr = np.vstack([scatter_low, scatter_high])
        ax.errorbar(x=x_values, y=median_constraint, yerr=yerr, lw=lw, color=colors_constraint[i_Poiss_frac], capsize=3,
                    marker="o", ms=4, markeredgewidth=1, elinewidth=lw, zorder=2)
        ax.axhline(Poiss_frac, color=colors_constraint[i_Poiss_frac], ls="--", lw=1, zorder=1)
        # if i_Poiss_frac == 0:
            # ax.text(0.1, Poiss_frac + 0.025, r"True $\eta_P$", color=colors_constraint[i_Poiss_frac], size="small")
    ax.set_xlabel("Expected counts per PS")
    ax.set_ylabel(r"Poisson flux fraction $\eta_P$")
    plt.tight_layout()
    fig.savefig(os.path.join(sys_data_path, out_file_plot), bbox_inches="tight")
