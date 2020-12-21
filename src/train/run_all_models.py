import subprocess
models = ["resnet50", "resnet34", "resnet18", "vgg", "mnasnet", "densenet", "basic"]

for model in models:
    subprocess.run(["python", "train.py", "/data2/cdcm",
     model, "/data2/cdcm/models", "--epochs", "5",
     "-v", "1", "--batched"] )