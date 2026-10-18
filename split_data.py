import os
import random
import tarfile

MASTER_TAR = os.getenv("BRATS_TAR", "./BraTS2021_Training_Data.tar")

# Define 3 separate hospital nodes and 1 global test set
NODE_A_DIR = "./data/node_A"
NODE_B_DIR = "./data/node_B"
NODE_C_DIR = "./data/node_C"
TEST_DIR = "./data/holdout_test"

def clean_patient_id(raw_path: str) -> str:
    norm = raw_path.replace("\\", "/").lstrip("./")
    parts = norm.split("/")
    if parts and parts[0].startswith("BraTS2021_") and not parts[0].endswith(".tar"):
        return parts[0]
    return ""

def setup_federated_nodes():
    if not os.path.exists(MASTER_TAR):
        print(f"❌ Cannot find master tar at: {MASTER_TAR}")
        return

    print("Opening master archive to index patient IDs...")
    with tarfile.open(MASTER_TAR, "r") as tar:
        all_members = tar.getmembers()
        patient_map = {}

        for m in all_members:
            pid = clean_patient_id(m.name)
            if pid:
                patient_map.setdefault(pid, []).append(m)

        all_pids = sorted(list(patient_map.keys()))
        print(f"Discovered {len(all_pids)} patient records.")

        # We need 30 patients per node (90 total) + 5 for test = 95 patients
        if len(all_pids) < 95:
            print(f"❌ Need at least 95 patients, found {len(all_pids)}.")
            return

        random.seed(42)
        random.shuffle(all_pids)
        
        # Distribute patients to nodes
        node_a_pids = all_pids[:30]
        node_b_pids = all_pids[30:60]
        node_c_pids = all_pids[60:90]
        test_pids = all_pids[90:95]

        directories = [NODE_A_DIR, NODE_B_DIR, NODE_C_DIR, TEST_DIR]
        for d in directories:
            os.makedirs(d, exist_ok=True)

        print(f"Extracting 30 cases for Hospital A...")
        tar.extractall(path=NODE_A_DIR, members=[m for pid in node_a_pids for m in patient_map[pid]])

        print(f"Extracting 30 cases for Hospital B...")
        tar.extractall(path=NODE_B_DIR, members=[m for pid in node_b_pids for m in patient_map[pid]])

        print(f"Extracting 30 cases for Hospital C...")
        tar.extractall(path=NODE_C_DIR, members=[m for pid in node_c_pids for m in patient_map[pid]])

        print(f"Extracting 5 holdout cases to {TEST_DIR}...")
        tar.extractall(path=TEST_DIR, members=[m for pid in test_pids for m in patient_map[pid]])

    print("\n✅ Multi-node extraction complete! 3 independent datasets are ready.")

if __name__ == "__main__":
    setup_federated_nodes()