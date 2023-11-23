import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
gradient_accumulation_steps = 2
epochs = 100
epochs_classifier = 50
max_steps = 500
num_classes = 2
num_channels = 8
gene_size = 18432
lr_classifier = 1e-3
lr_diffusion = 2e-4
num_of_samples = 1000
normalize_data = False
model_name = "PosSensitiveLarge"# "Unet"#"PosSensitiveLarge" #"PosSensitiveDeep"#"PosSensitive" # , "UnetLarge", ,
save_path = "syn_data_"+model_name

config = {
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "epochs": epochs,
    "epochs_classifier": epochs_classifier,
    "max_steps": max_steps,
    "num_classes": num_classes,
    "num_channels": num_channels,
    "gene_size": gene_size,
    "lr_classifier": lr_classifier,
    "lr_diffusion": lr_diffusion,
    "num_of_samples": num_of_samples,
    "save_path": save_path,
    "normalize_data": normalize_data,
    "model": model_name
}
