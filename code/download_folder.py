import wfdb
import os
import time
import sys
import contextlib
import requests
from tqdm import tqdm

# --- Monkey Patch requests to use a Browser User-Agent ---
# This is necessary because PhysioNet/servers may block default python scripts (403 Forbidden)
# even for public data. We simulate a Chrome browser.
_old_request = requests.Session.request

def _new_request(self, method, url, *args, **kwargs):
    headers = kwargs.get('headers', {})
    if 'User-Agent' not in headers:
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    kwargs['headers'] = headers
    return _old_request(self, method, url, *args, **kwargs)

requests.Session.request = _new_request
# ---------------------------------------------------------

# Configuration
PN_DIR = "mimic3wdb-matched"
LOCAL_DB_ROOT = r"e:\mimicIII\mimic3wdb-matched"
MASTER_RECORDS_FILE = os.path.join(LOCAL_DB_ROOT, "RECORDS")
DL_DIR = LOCAL_DB_ROOT
MAX_FOLDERS = 100

@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout and stderr from underlying libraries."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_folder_list(records_file, limit=None):
    folders = []
    if not os.path.exists(records_file):
        # We can't use standard print as it might be messy, but initialization errors are fine.
        print(f"Error: Master RECORDS file not found at {records_file}")
        return []
    
    with open(records_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if line.endswith('/'):
                    line = line[:-1]
                folders.append(line)
                if limit and len(folders) >= limit:
                    break
    return folders

def download_folder_contents(target_folder_path):
    # Use tqdm.write so it doesn't break any active outer bar (though overall bar is above)
    tqdm.write(f"Processing folder: {target_folder_path}")
    
    local_folder_full_path = os.path.join(DL_DIR, target_folder_path)
    os.makedirs(local_folder_full_path, exist_ok=True)

    records_file_url_path = f"{target_folder_path}/RECORDS"
    local_records_path = os.path.join(DL_DIR, records_file_url_path)
    
    # Download RECORDS silently
    try:
        with suppress_stdout():
            wfdb.dl_files(
                db=PN_DIR,
                dl_dir=DL_DIR,
                files=[records_file_url_path]
            )
    except Exception as e:
        tqdm.write(f"  Warning: Could not download RECORDS: {e}")
    
    sub_records = []
    if os.path.exists(local_records_path):
        with open(local_records_path, 'r') as f:
            for line in f:
                rec = line.strip()
                if rec:
                    full_rec = f"{target_folder_path}/{rec}"
                    sub_records.append(full_rec)
    else:
        tqdm.write(f"  No RECORDS file found, assuming single record.")
        sub_records.append(target_folder_path)

    # Use tqdm for progress. 'leave=False' makes it disappear after completion to keep screen clean,
    # or 'leave=True' to keep history. User said "timely clear screen", maybe they prefer transient logs.
    # But usually keeping history of "Processed X" is good. 
    # Let's align "规整一些" with "clean". 
    # I'll use leave=True but formatted nicely.
    # unit_scale=True creates cleaner numbers? No, records is count.
    
    pbar = tqdm(sub_records, desc=f"  Downloading", unit="rec", leave=True, position=1)
    
    for rec_path in pbar:
        # SKIP LOGIC
        rec_name = rec_path.split('/')[-1]
        hea_file = os.path.join(DL_DIR, rec_path + ".hea")
        
        if os.path.exists(hea_file):
            continue

        rec_start = time.time()
        try:
            with suppress_stdout():
                wfdb.dl_database(
                    PN_DIR,
                    DL_DIR,
                    records=[rec_path],
                    overwrite=True 
                )
            
            # Speed calc
            duration = time.time() - rec_start
            local_rec_dir = os.path.join(DL_DIR, os.path.dirname(rec_path))
            size_bytes = 0
            if os.path.exists(local_rec_dir):
                for f_name in os.listdir(local_rec_dir):
                    if f_name.startswith(rec_name):
                        size_bytes += os.path.getsize(os.path.join(local_rec_dir, f_name))
            
            speed_mb_s = (size_bytes / 1024 / 1024) / duration if duration > 0.1 else 0
            pbar.set_postfix_str(f"Speed: {speed_mb_s:.2f} MB/s")

        except Exception as e:
            tqdm.write(f"    ERROR detected for {rec_name}: {e}")
            pass

    pbar.close()

def main():
    # Only clear screen at the very start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"Targeting first {MAX_FOLDERS} folders.")
    folders = get_folder_list(MASTER_RECORDS_FILE, limit=MAX_FOLDERS)
    
    if not folders:
        print("No folders found.")
        return

    # position=0 ensures it sticks to top
    overall_pbar = tqdm(folders, desc="Overall Progress", unit="folder", position=0)
    for folder in overall_pbar:
        download_folder_contents(folder)
        
    print("\nAll downloads finished.")

if __name__ == "__main__":
    main()
