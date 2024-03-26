# Data generation

The script `make_dummy_dataset.py` is capable of generating the desired synthetic events for the NF studies in `pandas`-friendly `.csv`-format.
You should generate events both for "data" and for "MC".
We want to apply the normalising flows to morph MC into data.

## Explanation of the script

The script `make_dummy_dataset.py` can be used to create the benchmark toy dataset that we described and used in <input_arxiv_number> to test our single-flow correction architecture.
The dataset can be used to test correction methods for simulated events.
It comprises two or three kinematic variables (pT, eta, noise) and 2-N (we used four in our study) additional informative features, divided into two classes "A" (continuous, Gaussian-like) and "B" (discontinuous).

Please make sure that you have all the necessary libraries in your environment if you want to produce the toy dataset.
For your convenience, we provide an `environment.yml` in the `env` directory.

The script comes with a decent number of command-line arguments for flexibility.
You can check their meaning with `python make_dummy_dataset.py --help`.

The script requires config JSONs to steer the generation of the toy dataset, including parameter values and correlations.
We provide a set of example JSONs in the `configs` directory, which you can also take as blueprints for your dataset.

## Concrete commands

Execute the following two commands to generate your events in csv-format.

For "Data": `python make_dummy_dataset.py ./configs/viA_specs_data.json ./configs/viB_specs_data.json --pTScale 1.0 --etaSmear 0.25 --addNoise --finalCorrMatPath ./configs/final_corr_matrix_withNoise_data.json --nEvts 10000000 --seed 42 --outPath ./samples/my_data_withNoise.csv`

For "MC": `python make_dummy_dataset.py ./configs/viA_specs_MC.json ./configs/viB_specs_MC.json --pTScale 0.95 --etaSmear 0.2 --addNoise --finalCorrMatPath ./configs/final_corr_matrix_withNoise_MC.json --nEvts 10000000 --seed 43 --outPath ./samples/my_MC_withNoise.csv`

These two commands were the exact same ones that we used to generate the data used in the study of the single-flow architecture.

You can generate a few validation plots with `python plot_features.py <data/MC/together> --postfix <postfix>`, where the positional argument indicates the source of the histograms that you want to plot, and the postfix makes it more convenient to plot csv files with a postfix after `my_data` or `my_MC` in the filename.