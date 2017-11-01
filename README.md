# LNRSeeker 

A novel predicting tool for identifying Long Non-Coding RNAs by using deep learning.

## Prerequisite
1. Python 3.5 or later version 
2. numpy
3. pandas 
4. scikit-learn
5. keras with Tensorflow/Theano

We recommend installing Anaconda3 for setting up the runtime. 
  
## Usage 

To train the model, you may run
```{bash}
python LNRSeeker-train.py -c CODING_FILE -n NON_CODING_FILE -o OUTPUT_PREFIX
```

To use the model to predict, you may run
```{bash}
python LNRSeeker.py -m MODEL_FILE -i INPUT_FILE -o OUTPUT_PREFIX
```

## Useful Resources

We provide a default model saved in `LNRSeeker.pkl`, which was trained over
human transcripts from GENCODE and RefSeq.
The training data is also used by [PLEK](http://www.ibiomedical.net/plek/), where you may find [here](https://sourceforge.net/projects/plek/files/data_human/).





