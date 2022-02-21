import os
import glob
import pandas as pd

#Directory path is custom for myself. May vary depending on the storage location.
os.chdir("/Users/Kaspersky/Downloads/drive-download-20220218T092015Z-001")

file_extension = "csv"
all_filenames = [i for i in glob.glob('*.{}'.format(file_extension))]

#Combine all files in the list
combined_data = pd.concat([pd.read_csv(f) for f in all_filenames ])

#Export to csv
combined_data.to_csv("combined_data.csv", index=False, encoding='utf-8-sig')
