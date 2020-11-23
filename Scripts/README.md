# Scripts
* GCE_and_background_NN_predictions.npz
  > Contains 256 random maps and the corresponding NN predictions and true flux fractions of the testing data set from the proof-of-concept example (see Fig. 1 in the paper).
* GCE_and_background_settings.pickle
  > Contains the settings used for the NN that belongs the the maps in GCE_and_background_NN_predictions.npz.
* GCE_and_background_subset_256.pickle
  > Contains the 256 random maps and the corresponding flux fractions as a dictionary.
* analyse_simulated_uncertainties.py
  > This script can be used to simulate aleatoric and epistemic uncertainties and check that the coverage makes sense.
* augment_data_per_model.py
  > This script can be used for data augmentation (mirroring) of maps saved with "generate_data_per_model.py".
* check_flux_formula.py
  > This script can be used to check the scaling of the total point-source flux w.r.t. the NPTF parameters.
* check_training_maps.py
  > Iterate over all the training maps and print / save stats.
* combine_GCE_data.sh
  > Bash script that submits the PBS jobs for combining the data.
* combine_data_from_models.py
  > This combines data per model as generated with generate_data_per_model.py to a combined map and calculates the flux fractions.
* combine_data_from_models_for_SCD_test.py
  > This combines data per model as generated with generate_data_for_SCD_test.py to a combined map and calculates the flux fractions.
* create_GCE_data.sh
  > Bash script that submits the PBS jobs for generating the data.
* dN_dF_testing.py
  > This file can be used to compare the theoretically expected flux fractions to the actual flux fractions in the realisations.
* fit_Fermi_counts.py
  > This script performs an NPTFit for the Fermi counts or for mock data (see [NPTFit](https://github.com/bsafdi/NPTFit/tree/master/examples)).
* generate_best_fit_data.py
  > This script generates samples with best fit parameters for the GCE as determined by fit_Fermi_counts.py.
* generate_data_for_SCD_test.py
  > This script generates GCE maps for the SCD test.
* generate_data_per_model.py
  > This script generates maps for each model and saves map for each model separately. Then, they can be combined arbitrarily for the creation of training data.
* load_test_data.py
  > File that shows how to open the compressed data and obtain the maps.
* plot_accuracy_vs_training_samples.py
  > This script plots the mean accuracy as a function of the number of training samples used.
* plot_dN_dF.py
  > This script can be used to plot dN/dFs given by broken power laws (see NPTFit source code).
* plot_flux_fraction_posteriors_north_south.py
  > This script plots the flux fractions as estimated by the NNs trained on the northern and southern hemispheres separately
and on both hemispheres simultaneously.
* plot_templates.py
  > This script plots the Fermi templates.
* submit_NPTFit.pbs
  > PBS job script for running NPTFit.
* submit_combine_data_single.pbs
  > PBS job script for combining template maps for the generation of training / testing data.
* submit_data_gen_single.pbs
  > PBS job script for generating training / testing data.
