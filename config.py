import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
gradient_accumulation_steps = 16
epochs = 100
epochs_vae = 4000
epochs_classifier = 50
max_steps = 500
num_classes = 2
num_channels = 8
gene_size = 18432
lr_classifier = 1e-3
lr_diffusion = 2e-4
lr_vae = 6e-4
num_of_samples = 10000
normalize_data = True
kl_factor = 1e-2
gradient_clip = 1
channel_multiplier = (1,1,1,1,2,2,3,4)#(1,1,1,1,2,2,3,4,6,8,16,32) # (1,1,1,1,2,2,3,4)
attention_resolutions = [32,64]#[32,128] #[32,64]
base_width = 64
model_name = "UnetMLP"#"UnetMLP"# "Unet"#"PosSensitiveLarge" #"PosSensitiveDeep"#"PosSensitive" # , "UnetLarge", ,PosSensitive UnetMLP_working
save_path = "syn_data_"+model_name+str(attention_resolutions)+str(channel_multiplier)+str(base_width)
percent_unlabeled = 0#.95
label_proportion_for_generation = 0.3067

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
    "model": model_name,
    "kl_factor": kl_factor,
    "gradient_clip": gradient_clip,
    "channel_mult": channel_multiplier,
    "attention_resolutions": attention_resolutions,
    "base_width": base_width,
    "percent_unlabeled": percent_unlabeled,
    "label_proportion_for_generation": label_proportion_for_generation
}
