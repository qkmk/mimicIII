import wfdb
import os

# PhysioNet 上的相对路径（不含文件名）
pn_dir = "mimic3wdb-matched"

# 要下载的 record
record_name = "p00/p000020"

# 下载到当前目录
dl_dir = os.getcwd()

# 明确要下的文件类型
files = [
    f"{record_name}.hea",
    f"{record_name}.dat",
]

wfdb.dl_files(
    files=files,
    db=pn_dir,
    dl_dir=dl_dir
)

print("Download finished. Files in:", dl_dir)
