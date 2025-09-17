import os
import pandas as pd

# Get the directory this script is in
CURRENT_DIR = os.path.dirname(__file__)

# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Build the path to the data file
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_merged.parquet")

# Load the parquet file
df = pd.read_parquet(DATA_PATH)[
    [
        # identifiers / target
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        
        # product
        "ProductCD",
        
        # card info
        "card1", "card2", "card3", "card4", "card5", "card6",
        
        # address / distance
        "addr1", "addr2", "dist1", "dist2",
        
        # email
        "P_emaildomain", "R_emaildomain",
        
        # device
        "DeviceType", "DeviceInfo"
    ]
]