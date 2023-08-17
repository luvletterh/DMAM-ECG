# DMAM-ECG
 Diffusion Model with self-Attention Module for ECG Signal Denoising
 
The deep learning models were implemented using PyTorch.

# Dataset

~~~
bash ./data/download_data.sh
~~~


# Train the model
~~~
python main_exp.py
~~~



# Evaluation the model
~~~
python eval_new.py
~~~

## Acknowledgement

The data preprocessing is directly taken from [DeepFilter](https://www.sciencedirect.com/science/article/pii/S1746809421005899).

