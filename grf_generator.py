import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import qmc

def generate_grf_2d(size, alpha, tau, device='cpu'):
    """
    Gera um Gaussian Random Field (GRF) 2D com distribuição Matern usando FFT.
    alpha: Suavidade (rugosidade)
    tau: Comprimento de correlação (correlation length)
    """
    k_max = size // 2
    kx = torch.fft.fftfreq(size) * size
    ky = torch.fft.fftfreq(size) * size
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    
    k_sq = kx**2 + ky**2
    
    # Power spectrum (Matérn covariance in Fourier space)
    spectrum = 1.0 / (tau**2 + k_sq)**(alpha / 2.0)
    spectrum[0, 0] = 0.0 # Remove the mean
    
    # Random phase and amplitude
    noise = torch.randn(size, size, dtype=torch.cfloat)
    
    # Apply filter in frequency domain
    grf_fourier = noise * spectrum
    
    # Transform back to spatial domain
    grf_spatial = torch.fft.ifft2(grf_fourier).real
    
    # Normalize between -1 and 1
    grf_spatial = (grf_spatial - grf_spatial.min()) / (grf_spatial.max() - grf_spatial.min())
    grf_spatial = grf_spatial * 2.0 - 1.0
    
    return grf_spatial.to(device)

def generate_v3_dataset(num_samples=5000, grid_size=100, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Iniciando geração de {num_samples} amostras (GRFs) para a V3...")
    
    V_fields = torch.zeros(num_samples, grid_size, grid_size)
    
    # Parâmetros físicos variando (Latin Hypercube Sampling mock)
    params = torch.zeros(num_samples, 3) # b, kappa, u
    
    # 6 dimensões: alpha, tau, chance_parede, b, kappa, u
    sampler = qmc.LatinHypercube(d=6)
    lhs_sample = sampler.random(n=num_samples)
    
    # Mapeando os valores do hipercubo [0, 1] para os limites físicos
    l_bounds = [2.0, 1.0, 0.0, 0.5, 0.1, -1.0]
    u_bounds = [4.0, 10.0, 1.0, 2.0, 5.0, 1.0]
    lhs_scaled = qmc.scale(lhs_sample, l_bounds, u_bounds)
    
    for i in range(num_samples):
        alpha = lhs_scaled[i, 0]
        tau = lhs_scaled[i, 1]
        
        # Gera o campo de fundo
        grf = generate_grf_2d(grid_size, alpha, tau)
        
        # Adiciona barreiras / paredes (Máscaras) usando a variável LHS 2
        if lhs_scaled[i, 2] > 0.7:
            # wall_pos é um hiperparâmetro discreto aleatório extra (pode ser np random normal)
            wall_pos = np.random.randint(10, grid_size//2)
            grf[:wall_pos, :] = 10.0 # Parede infinita
            
        V_fields[i] = grf
        
        # Parâmetros Físicos
        params[i, 0] = lhs_scaled[i, 3]  # b (Kuhn)
        params[i, 1] = lhs_scaled[i, 4]  # kappa (Debye)
        params[i, 2] = lhs_scaled[i, 5]  # u (Flory-Huggins)
        
        if (i+1) % 500 == 0:
            print(f"  -> {i+1}/{num_samples} cenários espaciais gerados.")
            
    # Salva o Dataset
    save_path = os.path.join(output_dir, "pino_v3_grf_dataset.pt")
    torch.save({"V": V_fields, "params": params}, save_path)
    print(f"\n[+] Dataset salvo com sucesso em {save_path}!")
    
    # Salvar uma imagem de exemplo para visualização
    plt.imshow(V_fields[0].numpy(), cmap='inferno')
    plt.colorbar()
    plt.title("Exemplo de GRF Gerado (Amostra 0)")
    plt.savefig(os.path.join(output_dir, "exemplo_grf.png"))
    
if __name__ == "__main__":
    generate_v3_dataset(num_samples=5000)
