# Phanto-IDP: A model for precise and fast IDP backbone generation

![Phanto-IDP](./ImgSrc/Phanto-IDP.png)

Phanto-IDP is a VAE-based generative deep learning model, which aims to **precisely reconstruct** IDP conformation ensembles  and **generate** unseen structures **at a rather high speed**. This is a raw version of Phanto-IDP uploaded for further tests.



## Usage

### Environment Setup



### Generation with pre-trained model



### User-defined training process

For training model on your own trajectories, you have to first process the target trajectory into corresponding graph representation ensembles. This process is done with [mylddt](https://github.com/gjoni/mylddt) toolset, which accept one conformation at a time.

```shell
python pdb_parse.py
```

Then you can train the model with preset arguments in `arguments.py`, or adjust the parameters in command line as followed/

```shell
python main.py trial_run --epochs 400 --batch_size 32  
# the command line arguments should always start with your task name
```

One epoch on approximate 38,000 conformations may take ~130 seconds, and 400 epochs is absolutely enough for model convergence.