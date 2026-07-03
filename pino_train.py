import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from pino_architecture import PINO_Polyelectrolyte, sobolev_physics_loss

def train_pino_v3(epochs=1000, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Iniciando Treino Base do PINO V3 no dispositivo: {device}")
    
    # Inicializa a Arquitetura da V3
    model = PINO_Polyelectrolyte(modes1=16, modes2=16, width=96, input_channels=8).to(device) # 6 canais físicos + 2 canais espaciais (x, y)
    
    # Carrega o Dataset Gerado e Resolvido (GRF + SCFT)
    dataset_path = "data/pino_v3_dataset_complete.pt"
    if not os.path.exists(dataset_path):
        print("[!] Erro: Dataset não encontrado. Execute scft_solver.py primeiro.")
        return
        
    data = torch.load(dataset_path)
    V_fields = data["V"]
    params = data["params"]
    phi_scft = data["phi_scft"]
    
    dataset = TensorDataset(V_fields, params, phi_scft)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for V_batch, params_batch, phi_batch in dataloader:
            V_batch = V_batch.to(device).requires_grad_(True)
            phi_batch = phi_batch.to(device)
            
            # Montar canais: [V, x, y, b, kappa, u]
            grid_size = 100
            inputs = torch.zeros(V_batch.shape[0], grid_size, grid_size, 6, device=device)
            inputs[..., 0] = V_batch
            
            # Placeholder inputs
            b_plane = params_batch[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, grid_size, grid_size).to(device)
            inputs[..., 3] = b_plane
            
            optimizer.zero_grad()
            
            # Forward
            phi_pred = model(inputs).squeeze(-1)
            
            # Physics-Informed Sobolev Loss
            loss = sobolev_physics_loss(phi_pred, phi_batch, V_batch, model)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Época [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.6f}")
            
        # Treinamento real contínuo até o final das épocas

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/pino_v3_base.pth")
    print("\n[+] Treino Base Finalizado! Pesos salvos em weights/pino_v3_base.pth")

if __name__ == "__main__":
    train_pino_v3(epochs=1000)
