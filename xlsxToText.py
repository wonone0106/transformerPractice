from glob import glob
import pandas as pd

def xlsx_to_txt(input_folder, output_file):
    xlsx_paths = glob(f"{input_folder}/*.xlsx")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for path in xlsx_paths:
            df = pd.read_excel(path)

            for ko, en in zip(df["원문"], df["번역문"]):
                if isinstance(ko, str) and isinstance(en, str):
                    ko = ko.strip()
                    en = en.strip()
                    if ko and en:
                        f.write(ko + "\n")
                        f.write(en + "\n")

xlsx_to_txt("data/train", "data/train.txt")