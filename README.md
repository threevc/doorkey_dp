# Dynamic Programming for Minigrid Environment

## Important Code Files 
<ol>
<li> main.py
<li> trial1.ipynb
</ol>

## How to Run


```python
cd /path/to/dir/code 
python3 main.py ### Generates all gifs for the dynamic and the static environments
```
<!-- 
```python
cd /path/to/dir/code/icp_warm_up
python3 trial_icp.py ### Generates the post ICP visualizations of the various object point clouds
``` -->

## Dependencies 

A comprehensive list of dependencies is added in the attached `requirements.txt` file given in the folder `</path/to/dir/code>/code/starter_code/`. Some important packages are -
``` python
numpy
matplotlib
minigrid
imageio
```
To install the required dependencies, we can create a new mamba (or conda) environment named `<env>` .

``` bash
pip install -r requirements.txt ## Replace with conda if preferred
```
 <!-- To install jax with cuda support we can follow the command on the [jax install page](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier) -->

## Outputs

The file `main.py` creates gif visualization of the trajectories for the known environment and random environment cases.

The file `trial1.ipynb` does the same but in an interactive jupyter notebook for better control.

## File Structure


``` bash

.
├── code
│   └── starter_code
│       ├── create_env.py
│       ├── doorkey.py
│       ├── envs
│       ├── example.py
│       ├── gif
│            └── ## Result gifs from all environments
│       ├── main.py
│       ├── __pycache__
│       ├── README.md  ## Readme for the project problem statement
│       ├── requirements.txt
│       ├── trial1.ipynb
│       └── utils.py
├── ECE276B_PR1.pdf
├── environment.yml
└── README.md ## Output Readme for the work done


```