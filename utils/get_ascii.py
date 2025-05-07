import xml.etree.ElementTree as ET
import pandas as pd
import os
from dotenv import dotenv_values

values = dotenv_values()
ASCII_PATH = values['ASCII_PATH']

import pandas as pd

def get_ascii(code):
    filepath = os.path.join(ASCII_PATH, code.split("-")[0],code[:7],code+".txt")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # Find the line that starts with 'CSR:'
    try:
        csr_index = next(i for i, line in enumerate(lines) if line.strip() == "CSR:")
    except StopIteration:
        raise ValueError("CSR: section not found in file.")

    # Extract all lines after 'CSR:'
    csr_lines = lines[csr_index + 2:]

    # Build the DataFrame
    df = pd.DataFrame(enumerate(csr_lines), columns=["line", "text"])
    return df

# Example usage
if __name__ == "__main__":
    import os 
    from constants import TEST_RESULTS_PATH


    
    code = 'a01-000u'
    df = get_ascii(code)

    os.makedirs(TEST_RESULTS_PATH,exist_ok=True)
    save_path = os.path.join(TEST_RESULTS_PATH,"converted_ascii.csv")
    df.to_csv(save_path, index=False)
