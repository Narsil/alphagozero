# AlphaGo Zero

This is a unofficial reimplementation in Keras (Tensorflow backend) of [AlphaGo Zero]
It is work in progress. The aim is to have simple code.
Written in Python 3.

# Installation and run

Install the dependencies and devDependencies and start the server.

```sh
$ pip install -r requirements.txt # Requirements use tensorflow-gpu, feel free to use the CPU version.
$ python main.py
```

The main program will loop indefinitely, hopefully succeeding a building a stong model for Go play.
So far everything runs in one big loop. It might and up being split in the future in multiple threads/processes/workers.

If you want to play against a model to check its strength, you can use it's GTP support.
Just add the `gtp.py` file as an engine. Crude implementation for now.

# Configuration

Everythin is set in the `conf.py` file. All original papers parameters are described in the comments, but for a single machine, single GPU machine,
you should definitely stick to much lower parameters.

**Whenever you change configuration you should use reset.sh. But be careful, it will ERASE ALL PREVIOUS DATA (models and self play, and tensorboard logs)**



### Model

Everything is implemented in Keras and should correspond to the specified model in the Original paper. You can see the graph in tensorboard
```sh
python main.py  # The model is immediately created, so you can Ctrl+C pretty fast if you just want to check the graph
tensorboard --logdir=logs
```

### Self Play

Implementations should correspond to what is described in the paper.

### Train

I don't have 64 GPU workers as were used in the original implementation so for now, it just takes 64x the time than the original paper. Using Keras or TF to distribute the work is not planned. 

### Evaluator

The evaluator just runs games to check which model is best and saves the new model in best_model in case it ouranks the last best model


# Development

Want to contribute? Pull Requests more than welcome. 

TODOS:
 - Improve the tests (Right now basic gameplay functions are tested)
 - Find ways to distribute the work.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [AlphaGo Zero]: <https://deepmind.com/blog/alphago-zero-learning-scratch/>

