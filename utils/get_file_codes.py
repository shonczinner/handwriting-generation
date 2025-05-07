import os
from dotenv import dotenv_values

values = dotenv_values()
RAW_STROKES_PATH=values['RAW_STROKES_PATH']
ASCII_PATH=values['ASCII_PATH']

def get_stroke_codes():
    codes = []
    
    for folder0 in os.listdir(RAW_STROKES_PATH):
        folder0_path = os.path.join(RAW_STROKES_PATH, folder0)
        if not os.path.isdir(folder0_path):
            continue
        
        for folder1 in os.listdir(folder0_path):
            folder1_path = os.path.join(folder0_path, folder1)
            if not os.path.isdir(folder1_path):
                continue
            
            for file in os.listdir(folder1_path):
                file_path = os.path.join(folder1_path, file)
                if os.path.isfile(file_path):
                    filename_no_ext = os.path.splitext(file)[0]
                    stroke_code = "-".join(filename_no_ext.split("-")[:2])
                    codes.append(stroke_code)

    return codes


def get_ascii_codes():
    codes = []
    
    for folder0 in os.listdir(ASCII_PATH):
        folder0_path = os.path.join(ASCII_PATH, folder0)
        if not os.path.isdir(folder0_path):
            continue
        
        for folder1 in os.listdir(folder0_path):
            folder1_path = os.path.join(folder0_path, folder1)
            if not os.path.isdir(folder1_path):
                continue
            
            for file in os.listdir(folder1_path):
                file_path = os.path.join(folder1_path, file)
                if os.path.isfile(file_path):
                    filename_no_ext = os.path.splitext(file)[0]
                    stroke_code = "-".join(filename_no_ext.split("-")[:2])
                    codes.append(stroke_code)

    return codes

if __name__ == "__main__":
    import pandas as pd
    from constants import TEST_RESULTS_PATH

    stroke_codes = set(get_stroke_codes())
    ascii_codes = set(get_ascii_codes())
    
    all_codes = sorted(stroke_codes.union(ascii_codes))

    df = pd.DataFrame(index=all_codes)
    df['stroke'] = df.index.isin(stroke_codes)
    df['ascii'] = df.index.isin(ascii_codes)
    df['both'] = df['stroke'] & df['ascii']

    print(df.head())  # or save with df.to_csv("code_summary.csv")
    
    os.makedirs(TEST_RESULTS_PATH,exist_ok=True)
    save_path = os.path.join(TEST_RESULTS_PATH,"file_code_summary.csv")
    df.to_csv(save_path)
