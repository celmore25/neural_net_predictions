# neural_net_predictions
Class project for Machine Learning that will use neural networks to forecast energy market prices.

Path to final paper: "tex_writing/proposal_Elmore_Kopp.pdf

------------------------------------------------------------------------------------------------------------
Instructions on running neural network code:

Local running:

Using a conda evironment with scikit learn, keras, tensorflow, and numpy installed, run "python implementation/main_clay.py"
This will print the results for training a RNN at multiple locations on a rolling horizon.

CRC running:

With the same environment on a crc front end computer, run "qsub implementation/job.script"
This will run main_clay.py on a crc gpu (much faster)

Instructions on running graphing code:

All graphs are generated in the Jupyter notebook called "data_vis/graph_generator.ipynb"
"data_vis/scrape.py" is a python file used to scrape results generated by "main_clay.py" which is used in "data_vis/graph_generator.ipynb"

------------------------------------------------------------------------------------------------------------
A note on the dataset used:

The datset for this project is much to large to put into github. However, if you download the dataset provided by the students and place "all_prices.csv" into the "prices" direcetory, "main_clay.py" will be able to work. All additional files were used for experimental along the way of the project.

------------------------------------------------------------------------------------------------------------
List of directories and their contents:

data_vis: generating plots for final paper
evaluate: scraping output files
implementation: main directory for performing machine learning
long_run: auxiliary directory used to run longer tests on the crc
notebooks: starting jupyter notebooks to learn how to use tensorflow and keras (milestone 1)
prices: example price dataset (small examples for plotting)
tex_writing: directory that hodls all the files needed to write the final paper in latex. 

------------------------------------------------------------------------------------------------------------
A note on the latex bibliography used:

The .bib files used in the final paper are not included in the tex_writing directory because they were needed for a different local application. If access is needed to the bib files place contact Clay or Grace. However, the bibliography in the final paper is complete and accurate. 



Authors and Admin: Clay Elmore, Grace Kopp
