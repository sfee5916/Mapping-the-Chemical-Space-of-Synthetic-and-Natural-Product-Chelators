# Mapping-the-Chemical-Space-of-Synthetic-and-Natural-Product-Chelators
This repository holds the code used in the publication "Mapping the Chemical Space of Synthetic and Natural Product Chelators". The final dataset is available as a CSV within the "output_data" directory. Data are also available as a pickle file that can be loaded in python and stores the data as a dictionary with keys describing the data type and values as lists containing the values for each entry in the list. For an example of how to use this see the "2-PublicationImages.ipynb" jupyter notebook.

Coding was done using an anaconda environment, the yml file "SFChemicalSpace.yml" can be used to reproduce this.

Pickle file checkpoints that are generated before the final dataset are not included as well as the natural product datasets due to the size limitations. The checkpoint files are used when generating the final code and will be regenerated if the code is run. Natural product datasets are available as a release on this repository.
