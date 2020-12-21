# classify-modality
Craig Macfadyen Masters Dissertation Project at the University of St Andrews. Classifying medical images by modality over many datasets. 

## Usage

### Downloading TCIA Data
`src/tcia_download` contains the scripts required to download TCIA datasets en-masse.
 `get_collection_metadata.py` downloads metadata about each dataset hosted 
 by TCIA. These details are PatientIDs, Number of Images, Modalities 
 of images etc. This metadata is needed to run the download script. 
The Jupyter notebooks in `src/tcia_download/metadata` should then be fully 
run to process this metadata and create a train-validate-test 
split. Then the download script `tcia_download_script_sync.py`
can be run with the arguments "train", "validate", and "test" to 
download the data for each of the splits. **This will take a long time.**

### Preprocessing the Data
The data from TCIA is in .dcm format. This
is slow to load and has a lot of unnecessary metadata. 
`src/train/preprocessing.py` resizes and rescales 
all of the data and saves it in `.npy` format for ease of 
use in the training process. 


### Training the Networks
`src/train/train.py` contains a script for training 
deep CNNs on the created dataset. The other files 
are model declarations adapted from the models hosted 
on the PyTorch GitHub to take one channel images. 
`src/train/visualise_results.py` creates visualisations 
of the training metrics created in this process. 

### Testing the Networks
`src/test/` contains scripts for testing the networks. 
The training code saves the networks and the testing code 
loads the saved weights to evaluate their 
performance on the test set. `src/test/test_loader.py`
evaluates a given model on the test set and 
saves the results to be analysed.
`src/test/test_loader.py` does the visualisation and 
saves the images. 