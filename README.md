# One flow to correct them all

This repository holds the code for the "One flow to correct them all" project. Below, you can find some instructions.

## Enviroment

To set up the enviroment, one just have to run: 

```
conda env create -f enviroment.yaml
conda activate one-flow
```

## Replicating Results with 2D Datasets

In this section, we outline the steps required to replicate the results of our paper using two-dimensional datasets, starting with the transformation from the Checkerboard dataset to the Make Moons datasets.

### Generating and Training the Dataset

To generate and train the model with this dataset, use the script named `toy_Makemoons.py`. Running the script is straightforward and can be done by executing 

```
python toy_Make_moons.py
````

### Sample Generation

The script generates samples by employing `datasets_generation.py`, which is adapted from the flows4flows GitHub repository located at: [https://github.com/jraine/flows4flows/blob/main/ffflows/data/plane.py](https://github.com/jraine/flows4flows/blob/main/ffflows/data/plane.py).

After generating the samples, the datasets are divided into training, validation, and test sets. The test set is saved separately. Following this preparation, the training phase begins. Upon the completion of training, the model is saved, facilitating future evaluations and applications.

After completing the training, the results can be visualized using the test dataset. This involves running a specific script for plotting, which is tasked with generating plots based on the `Makemoons` dataset argument: 

```
python toy_2d_plotter.py --dataset Makemoons
```

The script will fetch the saved test dataset tensors along with the flow model from the training session and then proceed to generate the plots. The output plot is saved under the filename `Chessboard_makemoons.png`.

For reproducing the plots associated with the Four Circles datasets, the procedure mirrors that of the Make Moons dataset, necessitating a simple substitution of the dataset argument to `Fourcircles`.

# Reproducing Final Results with Two-Dimensional Datasets

In order to replicate the final results showcasing the flow's capability to learn and execute morphing between three distinct distributions, one can use the `three_way_flow.py` script. After the training is finished, one can produce the plots with 

```
python plot_three_way_results.py
```

# Physics-inspired dataset

To perform the generation of the `physics`-inspired dataset , one can look up the instructions inside the `data_generation` folder.

## Corrections

In order to perform the corrections (morph simulation into data) one can use the scripts inside the `simulation_correction_with_nl` folder. The first step is of course generate the samples. After that one can setup a shell script like the `run_FlowTraining.sh` to parse the needed information to the `main.py` script, which takes care of the training. 

After sourcing the `run_FlowTraining.sh` script, the training will begin and the code will run until completed. The own code will call the plotting functions to plot the marginal distirbutions.

The hyper-parameters of the flow can be changed in the `network_configs.yaml`, a couple of example are already inside.

## Correlation matrices, ROC curve and profile plots

To perform the reminaing plots from the paper, the codes inside the `simulation_correction_with_nl/classifier_studies` should be used. Again, one should input the nescessary paths for the `flow_vs_data_BDT.py` which should take care of the rest. It will train a BDT to distinguish betwenn flow corrected toy simulation and toy data and produce the prodeces the BDT output and profiles plots based on that. After that another BDT will be trained in the toy simulation x toy data and the ROC curve will be compared betwenn these two results, showcasing the performance of the one flow corrections.
