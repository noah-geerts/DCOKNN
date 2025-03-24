# Prerequisites

- Python == 3.8, numpy == 1.20.3, faiss

# Datasets

The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html.

1. Download the dataset GIST from ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz. The dataset is large. This step may take several minutes. The data format can be found in http://corpus-texmex.irisa.fr/.

   ```shell
   wget -O ./gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz --no-check-certificate
   ```

   ```windows powershell
   curl -o ./gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
   ```

2. Unzip the dataset.

   ```shell
   tar -zxvf ./gist.tar.gz -C ./
   ```

   ```windows powershell
   tar -zxvf .\gist.tar.gz -C .\
   ```

3. Optional: rename the gist data file (or other dataset) to [dataset name]\_raw,
   and run fvec.py to reduce the number of dataset vectors 100x and number
   of query vectors 10x. This can be used for setting up the repository
   so that everything runs more quickly.

4. Preprocess the dataset with random orthogonal transformation.

   ```shell
   python randomize.py
   ```

5. Preprocess the dataset with PCA transformation.

   ```shell
   python randomize.py
   ```

6. Generate the clustering of the dataset for IVF.

   ```shell
   python ivf.py
   ```
