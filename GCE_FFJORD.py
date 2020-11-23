import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfb = tfp.bijectors
tfd = tfp.distributions

# TODO: implement FFJORD!
# MLP ODE class
class MLP_ODE(tf.Module):
    """Multi-layer NN ode_fn."""
    def __init__(self, num_hidden, num_layers, num_output, name='mlp_ode', activation="softplus",
                 activation_last="linear", latent_vector=None):
        super(MLP_ODE, self).__init__(name=name)
        self._num_hidden = num_hidden
        self._num_output = num_output
        self._num_layers = num_layers
        self._modules = []
        self._latent = latent_vector
        for i_layer in range(self._num_layers - 1):
            self._modules.append(tf.keras.layers.Dense(self._num_hidden, activation=activation))
        self._modules.append(tf.keras.layers.Dense(self._num_output, activation=activation_last))
        self._model = tf.keras.Sequential(self._modules)

    def __call__(self, t, inputs):
        original_shape = inputs.shape
        inputs = tf.concat([tf.broadcast_to(t, list(inputs.shape)[0:2] + [1]), inputs], -1)
        if self._latent is not None:
            inputs = tf.concat([inputs, tf.tile(tf.expand_dims(self._latent, 0), [inputs.shape[0], 1, 1])], -1)
        # Flatten
        inputs = tf.reshape(inputs, [np.multiply(*original_shape[0:2]), -1])
        # Run NN
        model_output = self._model(inputs)
        # Reshape to n_points x n_batch x n_templates
        model_reshaped = tf.reshape(model_output, original_shape)
        return model_reshaped


# GCE FFJORD CLASS
class GCE_FFJORD:

    def __init__(self, op_logits_dict, n_params, latent_vector, stacked_ffjords=4, n_hidden_ffjord=8, n_layers_ffjord=3,
                 solver_tol_ffjord=1e-5, trace_exact_ffjord=False, std_if_not_given_ffjord=0.1,
                 n_sample_ffjord=64, activation_ffjord="softplus", activation_last_ffjord="sigmoid"):

        if "covar" in op_logits_dict.keys():
            raise NotImplementedError("FFJORD based off full Gaussian covariance has not been implemented yet!")
        elif "logvar" in op_logits_dict.keys():
            self._base_distribution = tfd.MultivariateNormalDiag(loc=op_logits_dict["logits_mean"], scale_diag=tf.exp(0.5 * op_logits_dict["logvar"]))
        else:
            print("No Gaussian uncertainties were found to base FFJORD on. Using the default initial STD of", str(std_if_not_given_ffjord) + ".")
            scale_diag = std_if_not_given_ffjord * np.ones(list(op_logits_dict["logits_mean"].shape)).astype(np.float32)
            self._base_distribution = tfd.MultivariateNormalDiag(loc=op_logits_dict["logits_mean"], scale_diag=scale_diag)

        # Building bijector
        solver = tfp.math.ode.DormandPrince(atol=solver_tol_ffjord)
        ode_solve_fn = solver.solve
        # TODO: hutchinson for training, exact for evaluation?
        trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact if trace_exact_ffjord else tfb.ffjord.trace_jacobian_hutchinson

        bijectors = []
        for _ in range(stacked_ffjords):
            mlp_model = MLP_ODE(n_hidden_ffjord, n_layers_ffjord, n_params, activation=activation_ffjord,
                                activation_last=activation_last_ffjord, latent_vector=latent_vector)
            next_ffjord = tfb.FFJORD(state_time_derivative_fn=mlp_model, ode_solve_fn=ode_solve_fn,
                                     trace_augmentation_fn=trace_augmentation_fn)
            bijectors.append(next_ffjord)

        self._stacked_ffjord = tfb.Chain(bijectors[::-1])
        self.transformed_distribution = tfd.TransformedDistribution(distribution=self._base_distribution,
                                                                    bijector=self._stacked_ffjord)
        self.op_sample = self.get_samples(n_sample_ffjord)

    def get_samples(self, n_samples):
        base_distribution_samples = self._base_distribution.sample(n_samples)
        transformed_samples = self._stacked_ffjord.forward(base_distribution_samples)
        # transformed_samples = self.transformed_distribution.sample(n_samples)
        return base_distribution_samples, transformed_samples

    def get_base_params(self):
        return self._base_distribution.loc, self._base_distribution.scale

