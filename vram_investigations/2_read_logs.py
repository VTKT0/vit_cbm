# synthetic_runs.py creates example data, which we can then extract from the log file to create charts

import pandas as pd
import matplotlib.pyplot as plt


with open("batch.log", "r") as f:
    log_contents = f.read()



# Split log into lines
log_lines = log_contents.strip().split('\n')

rows = []
run_name = None


for line in log_lines:

    ts = line.split("|")[0].strip()
    info = line.split("|")[1].strip()
    data = [x.strip() for x in line.split("|")[2:]]

    # when we start a new run, get the info for it
    if data[0][0] == '#':
        run_name = data[0].split("#")[1]
        params = [(x.split("=")[0].strip(), x.split("=")[1].strip()) for x in data[0].split("#")[2:]]
        run_info = {'run_name':run_name}
        for param in params:
            run_info[param[0]] = param[1]

    elif len(data) > 1:
        fields = [(x.split("=")[0].strip(), x.split("=")[1].strip()) for x in data]

        row = {'ts':ts}

        for key, value in run_info.items():
            row[key] = value


        for field in fields:
            row[field[0]] = field[1].replace("GB", "")
    


        rows.append(row)

df = pd.DataFrame(rows, columns=rows[0].keys())

df.to_csv("memory_charts_batch_limit.csv", index=False)

