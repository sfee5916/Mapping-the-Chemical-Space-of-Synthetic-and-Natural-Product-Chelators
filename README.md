# Mapping-the-Chemical-Space-of-Synthetic-and-Natural-Product-Chelators
This repository holds the code used in the paper "Mapping the Chemical Space of Synthetic and Natural Product Chelators". The final datasets is available as a CSV within the "output_data" directory. Data is also available as a pickle file that can be loaded in python and stores the data as a dictionary with keys describing the data type and values as lists containing the values for each entry in the list. For an example of how to use this see the "2-PublicationImages.ipynb" jupyter notebook.

Coding was done using an anaconda environment, the yml file "SFChemicalSpace.yml" can be used to reproduce this.

Pickle file checkpoints that are generated before the final dataset are not included as well as the natural product datasets due to the size limitations on Github. The checkpoint files are used when generating the final code and will be regenerated if the code is ran. Natural product datasets are available as a release on this repository.

"supernatural.csv" and "coconut_csv-05-2025.csv" have been compressed using 7zip to allow them to be added to the repository. Input data was supplied to allow reproduction of the dataset generated.
