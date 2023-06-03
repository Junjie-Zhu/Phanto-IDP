import os
import sys

all_dir = sys.argv[1]
fwrite = open("pdb_list.dat", "w")
for files in os.listdir(all_dir):
    if files.endswith(".pdb"):
        fwrite.write("%s\n" % files)

fwrite.close()

