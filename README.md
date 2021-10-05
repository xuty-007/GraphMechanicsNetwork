# Graph Mechanics Networks

#### Environment

The Python environment is depicted in `requirements.yml`.



#### Data preparation

Our simulation scripts are placed under `spatial_graph/n_body_system/dataset`.

To generate datasets containing multiple isolated particles, sticks, and hinges, use the following command under the path `spatial_graph/n_body_system/dataset`

```bash
python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 3 --n_stick 2 --n_hinge 1 --n_workers 50
```

where the arguments `n_isolated`, `n_stick`, and `n_hinge` indicate the number of isolated particles, sticks, and hinges, respectively. The argument `n_workers` refers to the number of parallel threads for parallel data generation. For other potential arguments, please refer to `generate_dataset.py`.

By default, the generated data will be placed in a new folder named `data` under the current path.



#### Model training and evaluation

Under the root path, simply use

```bash
python -u spatial_graph/main.py --config_by_file --outf logs 2>&1 | tee out.log
```

where the `--config_by_file` option enables loading the hyper-parameters from the file `simple_config.json`.

To run experiments under different scenarios, simply change the hyper-parameters in `simple_config.json`. For instance, one may change the `n_isolated`, `n_stick`, and `n_hinge` configuration to test the model with various object combinations.



For more details, please refer to our paper.
