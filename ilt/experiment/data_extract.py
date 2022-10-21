import os

case_results = sorted(os.listdir("."))
print(case_results)
# pvband = []
# loss = []
# epe = []
for case_result in case_results:
    with open(case_result) as f:
        pvbands = []
        losses = []
        epes = []
        data_results = f.readlines()
        data_results = data_results[1:]
        for i in range(1, len(data_results), 4):
            start_index = data_results[i].rfind(":")
            pvbands.append(data_results[i][start_index+2:-1])
        for i in range(2, len(data_results), 4):
            start_index = data_results[i].rfind(":")
            epes.append(data_results[i][start_index+2:-1])
        for i in range(3, len(data_results), 4):
            start_index = data_results[i].rfind(":")
            losses.append(data_results[i][start_index + 2:-1])
    with open(case_result + "_pvband.txt", 'w') as f:
        for pvband in pvbands:
            f.write(str(pvband) + "\n")
    with open(case_result + "_loss.txt", 'w') as f:
        for loss in losses:
            f.write(str(loss) + "\n")
    with open(case_result + "_epe.txt", 'w') as f:
        for epe in epes:
            f.write(str(epe) + "\n")
