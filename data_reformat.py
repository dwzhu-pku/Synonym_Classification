import csv
PATH = "./Datasets/part01"

def data_reformat(split: str):
    with open(f"{PATH}/{split}.txt", 'r', encoding="utf-8") as fin, open(f"{PATH}/{split}.csv", 'w', encoding="utf-8", newline='') as csvfile:
        line_list = list()
        for line in fin.readlines():
            line_list.append(line.split('\t'))
        print(f"#lines of {split}: {len(line_list)}")

        fieldnames = ['text1', 'text2', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for line in line_list:
            writer.writerow({'text1': line[0].strip(), 'text2': line[1].strip(), 'label': line[2].strip()})


data_reformat("train")
data_reformat("valid")
data_reformat("test")


        
        
