import torch
import os
import gc

def augment_dataset(input_path="data/pino_v3_dataset_complete.pt", output_path="data/pino_v3_dataset_augmented.pt"):
    print("[*] Carregando dataset original para Augmentation...")
    if not os.path.exists(input_path):
        print("[!] Erro: Dataset não encontrado!")
        return

    data = torch.load(input_path)
    V_fields = data["V"]        # Shape: (N, 100, 100)
    phi_scft = data["phi_scft"] # Shape: (N, 100, 100)
    params = data["params"]     # Shape: (N, 3)

    num_samples = V_fields.shape[0]
    
    print(f"[*] Dataset original: {num_samples} amostras.")
    print("[*] Aplicando simetrias geométricas (Grupo D4)...")
    
    # Listas para armazenar as variações
    aug_V = []
    aug_phi = []
    aug_params = []
    
    for k in range(4):
        # Rotações: 0, 90, 180, 270
        V_rot = torch.rot90(V_fields, k, dims=[1, 2])
        phi_rot = torch.rot90(phi_scft, k, dims=[1, 2])
        
        aug_V.append(V_rot)
        aug_phi.append(phi_rot)
        aug_params.append(params)
        
        # Espelhamento (Flip Horizontal) das rotações
        V_flip = torch.flip(V_rot, dims=[2])
        phi_flip = torch.flip(phi_rot, dims=[2])
        
        aug_V.append(V_flip)
        aug_phi.append(phi_flip)
        aug_params.append(params)
        
    print("[*] Concatenando os tensores no formato expandido (8x)...")
    V_expanded = torch.cat(aug_V, dim=0)
    phi_expanded = torch.cat(aug_phi, dim=0)
    params_expanded = torch.cat(aug_params, dim=0)
    
    total_samples = V_expanded.shape[0]
    print(f"[+] Dataset final expandido para {total_samples} amostras!")
    
    print(f"[*] Salvando novo dataset gigante em {output_path} ... (isso pode demorar um pouco)")
    torch.save({
        "V": V_expanded,
        "phi_scft": phi_expanded,
        "params": params_expanded
    }, output_path)
    
    print("[+] Salvamento concluído com sucesso!")

if __name__ == "__main__":
    augment_dataset()
