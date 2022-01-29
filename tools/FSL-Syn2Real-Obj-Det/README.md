### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# 1. Create training set
python createTrainingDataset.py --google_scenes_data <path-1.1> --syn_data_loc <path-2.1> --out_dir <path-3.1>

# 2. Create test set
python createTestDataset.py --ocid_data_loc <path-1.2> --real_obj_data_loc <path-2.2> --out_dir <path-3.2>

# 3. move path-3.1 and path-3.2 to a new directory, say <path>

# 4. After step 3, compress using 7zip
7z a <name>.7z <path> 
```

### Download and extract the dataset
```bash
wget <dataset-url> #download
7za x FSL-Sim2Real-IRVL-2022.7z # decompress
```