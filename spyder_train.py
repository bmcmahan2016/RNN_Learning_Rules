# run the following in a spyer console to train networks
for num in range(10):
    model_name = "bptt_624" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 1")
    model_name = "ga_624" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 1")

for num in range(10):
    model_name = "bptt_724" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 2")
    model_name = "ga_724" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 2")

for num in range(10):
    model_name = "bptt_824" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 3")
    model_name = "ga_824" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 3")

for num in range(10):
    model_name = "bptt_924" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 4")
    model_name = "ga_924" + str(num)
    runfile("train_models.py", model_name+" -v 0.5 --N 4")