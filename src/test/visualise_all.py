from visualise_test_results import visualise_model_results

model_info = [
    ("resnet18-0", "ResNet18", True),
    ("resnet34-0", "ResNet34", True),
    ("resnet50-0", "ResNet50", True),
    ("vgg-0", "VGG16", True),
    ("basic-0", "Simple CNN", True),
    ("mnasnet-0", "MNASNet", True),
    ("densenet-0", "DenseNet", True),
    ("vgg_dropout-0", "VGG with Dropout Layers", False),
    ("vgg_dropoutw-0.005-0", "VGG with Dropout Layers and Weight Decay = 0.005", False),
    ("vgg_dropoutw-0.01-0", "VGG with Dropout Layers and Weight Decay = 0.01", False),
    ("resnet18-4", "ResNet18", True),
    ("resnet34-4", "ResNet34", True),
    ("resnet50-4", "ResNet50", True),
    ("vgg-4", "VGG16", True),
    ("basic-4", "Simple CNN", True),
    ("mnasnet-4", "MNASNet", True),
    ("densenet-4", "DenseNet", True),
    ("vgg_dropout-4", "VGG with Dropout Layers", False),
    ("vgg_dropoutw-0.005-4", "VGG with Dropout Layers and Weight Decay = 0.005", False),
    ("vgg_dropoutw-0.01-4", "VGG with Dropout Layers and Weight Decay = 0.1", False),
    ("vgg_dropoutw-0.02-3", "VGG with Dropout Layers and Weight Decay = 0.2", False),
]

for info in model_info:
    visualise_model_results("test", *info)
