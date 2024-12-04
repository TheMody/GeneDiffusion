import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
gradient_accumulation_steps = 8
epochs = 100
epochs_vae = 4000
epochs_classifier = 10
max_steps = 500
num_classes = 26
num_channels = 8
gene_size = 26624
lr_classifier = 5e-5
lr_pretrain = 1e-4
lr_diffusion = 2e-4
lr_vae = 1e-4
num_of_samples = 2504
normalize_data = True
kl_factor = 1e-1
gradient_clip = 1
mask_ratio = 0.7
gradient_clipping = 1.0
test_set_size = 1040
enforce_zeros = True
zero_mask = torch.load("data/zero_mask1k.pt")
channel_multiplier = (1,1,1,1,2,2,3,4)#(1,1,1,1,2,2,3,4,6,8,16,32) # (1,1,1,1,2,2,3,4)
attention_resolutions = [32,64]#[32,128] #[32,64]
base_width = 64
model_name = "UnetCombined" # "UnetMLP"# "Unet"#"PosSensitiveLarge" #"PosSensitiveDeep"#"PosSensitive" # , "UnetLarge", ,PosSensitive UnetMLP_working #Baseline
save_path = "syn_data_1k_"+model_name
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
    "lr_pretrain": lr_pretrain,
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
    "label_proportion_for_generation": label_proportion_for_generation,
    "gradient_clipping": gradient_clipping,
    "test_set_size": test_set_size
}
