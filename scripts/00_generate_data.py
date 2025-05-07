import pandas as pd
import os

from utils.get_file_codes import get_stroke_codes, get_ascii_codes
from utils.parse_strokes import parse_strokes
from utils.get_ascii import get_ascii

# gets stroke and ascii codes, generates stroke and ascii datasets
# stroke has columns: code, line, delta_x, delta_y, lift_point and is assumed to be in order
# ascii has columns: code, line, text
# note that ascii codes and stroke codes are not always the same
# therefore care must be taken if trying to make a combined dataset of strokes and ascii
def generate_data():
    stroke_codes = set(get_stroke_codes())
    stroke_dfs = []

    print("Parsing stroke codes...")
    for i, code in enumerate(stroke_codes, 1):
        print(f"[{i}/{len(stroke_codes)}] Processing stroke code: {code}")
        df = parse_strokes(code)
        df['code'] = code
        stroke_dfs.append(df)

    stroke_df = pd.concat(stroke_dfs, ignore_index=True)

    ascii_codes = set(get_ascii_codes())
    ascii_dfs = []

    print("Parsing ASCII codes...")
    for i, code in enumerate(ascii_codes, 1):
        print(f"[{i}/{len(ascii_codes)}] Processing ASCII code: {code}")
        df = get_ascii(code)
        df['code'] = code
        ascii_dfs.append(df)

    ascii_df = pd.concat(ascii_dfs, ignore_index=True)

    return stroke_df, ascii_df



if __name__=="__main__":
    from constants import PROCESSED_PATH

    stroke_df, ascii_df = generate_data()

    os.makedirs(PROCESSED_PATH,exist_ok=True)

    strokes_save_path = os.path.join(PROCESSED_PATH,"strokes_data.csv")
    stroke_df.to_csv(strokes_save_path)

    ascii_save_path = os.path.join(PROCESSED_PATH,"ascii_data.csv")
    ascii_df.to_csv(ascii_save_path)


