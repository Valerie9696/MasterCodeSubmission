Welcome!

This is the code submission of my master thesis.
The folder DDxT contains all files related to the DDxT model, while the folder This contains all files related
to the DDxT model with altered positional encodings. These were originally two separate git projects (to prevent mix-ups), so some code may double.
It is only fused for this submission.

In both folders is a file called config.py. In this file, please set the path of the variable DATA_DIR to wherever you want to save the model outputs (the directories will be automatically set up based on this root directory).
Download the dataset from https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585
Rename the following files:
release_conditions.json -> conditions.json
release_evidences.json -> evidences.json
release_test_patients.csv -> test.csv
release_train_patients.csv -> train.csv
Put these files inside the directory, specified in DATA_DIR.

Install the dependencies from the requirements.txt
Additionally, Graphviz needs to be installed (as a program and a python package) via this link: https://graphviz.org/download/

To recreate the results of this thesis:

1. Either leave the version_name in config.py as is (final_full) and go to Step 2, or rename version_name and run the training using main.py. Not renaming version_name will overwrite my checkpoints in the checkpoints folder.
2. Run test.py to get the plot of the loss, Accuracy, F1 score, confusion matrix, and table with metrics per primary diagnosis.
3. Run clinical_pictures.py to generate the plots of average age and sex per diagnosis and make a dictionary of evidences ordered by importance per diagnosis.
4. Run explanation_charts.py to generate the explanation charts used for SoSci survey. Uncomment l.821 to show in browser (these are only the text columns and plots of bar charts and pie charts. The full charts wit graphs were assembled in SosSi survey and can be found in the xml file of the study within the root directory of this project.). The graphs can be found in DATA_DIR/Plots/<version_name>/full.

For the baseline:

The DDxT folder contains mostly original files from the repository https://github.com/MahmudulAlam/Differential-Diagnosis-Using-Transformers with some adaptions from my version for the explanations.

Again, please set the DATA_DIR variable in the config.py file to the path of the root directory for the results of DDxT.

Put the data from the link above in DDxT/data and leave the names as the original ones.

Either leave the version_name in config.py as is (final_full) and go to Step 2, or rename version_name and run the training using main.py. Not renaming version_name will overwrite my weights in the weights folder.
Then repeat steps 2 to 4.

To recreate the Wilcoxon rank-sum and MLR test results, please run survey_evaluation.py. If you want to see the results for the assumption tests,
you need to change line 214 to True and test_assumptions=True in line 630.

Please note, that the survey data was filtered by the Prolific IDs of participants, which had to be deleted as it is personal information of participants.
Thus, the code which initially filtered the full dataset with IDs is commented out and only the filtered version with 64 participants can be loaded.
The xml file of the SoSci survey project, its structure without content, and the code_book that explains the codes in the filtered_survey_results.csv are provided in the root of this project as survey.xml, structure.csv, and code_book.xlsx