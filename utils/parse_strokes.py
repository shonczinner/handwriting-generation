import xml.etree.ElementTree as ET
import pandas as pd
import os
from dotenv import dotenv_values

values = dotenv_values()
RAW_STROKES_PATH = values['RAW_STROKES_PATH']

def parse_strokes(code):
    xml_folder = os.path.join(RAW_STROKES_PATH, code.split("-")[0],code[:7])
    files  = [file for file in os.listdir(xml_folder) if file.startswith(code) and file.endswith(".xml")]
    
    rows = []
    for line,file in enumerate(files):

        xml_file = os.path.join(xml_folder, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        prev_x = prev_y = None  # Reset per file

        for stroke in root.findall(".//Stroke"):
            points = stroke.findall("Point")

            if not points:
                continue

            for i, point in enumerate(points):
                x = int(point.get("x"))
                y = int(point.get("y"))

                if prev_x is None:
                    delta_x, delta_y = 0, 0
                else:
                    delta_x = x - prev_x
                    delta_y = y - prev_y

                last_point = 1 if i == len(points) - 1 else 0
                rows.append((line,delta_x, delta_y, last_point))

                prev_x, prev_y = x, y

    df = pd.DataFrame(rows, columns=["line","delta_x", "delta_y", "lift_point"])
    return df

# Example usage
if __name__ == "__main__":
    import os 
    from constants import TEST_RESULTS_PATH


    
    code = 'a01-000u'
    df = parse_strokes(code)

    os.makedirs(TEST_RESULTS_PATH,exist_ok=True)
    save_path = os.path.join(TEST_RESULTS_PATH,"converted_output.csv")
    df.to_csv(save_path, index=False)
