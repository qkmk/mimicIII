import wfdb
import os

pn_dir = "mimic3wdb-matched"
dl_dir = r"e:\mimicIII\mimic3wdb-matched"
rec_path = "p00/p000052/3238451_0001"

print(f"Attempting to download {rec_path}...")
try:
    wfdb.dl_database(pn_dir, dl_dir, records=[rec_path], overwrite=True)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
