import torch
import os
import matplotlib.pyplot as plt

def run_scft_solver_mock(dataset_path="data/pino_v3_grf_dataset.pt"):
    """
    Simulador SCFT (Mock): Recebe os potenciais GRF e resolve a densidade polimérica real.
    Na vida real, este script chamaria um binário em C++/Fortran (ex: pseudo-spectral solver).
    Aqui vamos gerar uma densidade pseudo-física aproximada para prosseguir com o pipeline.
    """
    print("[*] Iniciando Solucionador SCFT (Self-Consistent Field Theory)...")
    if not os.path.exists(dataset_path):
        print("[!] Erro: Dataset GRF não encontrado.")
        return
        
    data = torch.load(dataset_path)
    V_fields = data["V"]
    params = data["params"]
    num_samples = V_fields.shape[0]
    grid_size = V_fields.shape[1]
    
    # Criaremos o tensor da densidade real "Ground Truth"
    phi_scft = torch.zeros(num_samples, grid_size, grid_size)
    
    print(f"[*] Resolvendo {num_samples} cenários...")
    for i in range(num_samples):
        # A densidade tende a se concentrar onde o potencial é menor (distribuição de Boltzmann)
        # phi ~ exp(-V / kT)
        V = V_fields[i]
        
        # Simulação mock de relaxação
        phi = torch.exp(-V * 2.0) 
        
        # Zera onde tem parede infinita (V = 10)
        phi[V > 9.0] = 0.0
        
        # Normalização (conservação de massa)
        phi = phi / (torch.sum(phi) + 1e-8) * 100.0 # Massa total arbitrária
        
        phi_scft[i] = phi
        
        if (i+1) % 500 == 0:
            print(f"  -> {i+1}/{num_samples} resolvidas pelo SCFT.")
            
    # Salva o Dataset Completo (V + Params + Phi_True)
    out_path = "data/pino_v3_dataset_complete.pt"
    torch.save({
        "V": V_fields,
        "params": params,
        "phi_scft": phi_scft
    }, out_path)
    print(f"\n[+] Dataset Ancorado Final gerado com sucesso em {out_path}!")
    
    # Salvar imagem comparativa
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(V_fields[0].numpy(), cmap='inferno')
    axes[0].set_title('Potencial (GRF)')
    axes[1].imshow(phi_scft[0].numpy(), cmap='viridis')
    axes[1].set_title('Densidade Polimérica (SCFT)')
    plt.savefig("data/exemplo_scft_solution.png")
    plt.close()

if __name__ == "__main__":
    run_scft_solver_mock()
