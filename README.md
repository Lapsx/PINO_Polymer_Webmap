# PINO Polymer Webmap (V3)

## Objetivo do Projeto

O **PINO Polymer Webmap** é a evolução definitiva (V3) da nossa arquitetura de predição de densidades poliméricas. Nascido como uma Fourier Neural Operator (FNO) clássica em versões anteriores, o projeto tem como objetivo principal **transcender o paradigma de mapeamento puramente estatístico (dados para dados) em direção a um aprendizado profundo guiado pelas leis fundamentais da física termodinâmica**.

A base teórica e de engenharia desta V3 introduz o **Physics-Informed Neural Operator (PINO)**, almejando criar um modelo capaz de generalizar cenários de confinamento e interação eletrostática sobre polieletrólitos sem depender exclusivamente de simulações exaustivas de Self-Consistent Field Theory (SCFT).

Os principais pilares teóricos discutidos que sustentam este objetivo são:
1. **Representação Contínua da Carga**: Utilização de coeficientes de Fourier para modelar sequências de carga do polímero, comprimindo um espaço combinatório colossal para um espaço de frequência reduzido e interpretável.
2. **Diversidade Topológica Infinita (O Fator Surpresa)**: Substituição de cenários rígidos de ancoragem por *Gaussian Random Fields (GRFs)* com parâmetros Matern para simular as mais extremas e caóticas topologias de confinamento e potenciais. O espaço paramétrico que dita a topologia desses campos e as propriedades termodinâmicas do polímero é amostrado usando **Latin Hypercube Sampling (LHS)**, garantindo cobertura estatística perfeita sem viés.
3. **Exploração da Simetria Física (Data Augmentation D4)**: Pelo fato da física de matrizes de confinamento e a resposta polimérica ($\phi$) serem espacialmente isotrópicas e invariantes a rotações e espelhamentos, aplica-se o Grupo de Simetria D4 (rotações 90º, 180º, 270º e reflexões) offline para multiplicar exponencialmente o *Ground Truth* do SCFT sem custo computacional, mitigando completamente o *overfitting*.
4. **Treinamento de Sobolev (O Chute Analítico)**: Injeção das derivadas do modelo na Função de Custo (*Loss*). A rede não aprende apenas a densidade $\phi(\mathbf{r})$, mas também a sua suscetibilidade térmica exata $\left( \frac{\delta \phi}{\delta V} \right)$, multiplicando a riqueza informacional de cada simulação.
5. **Resíduos Físicos & Aprendizado Ativo**: Utilização de dezenas de milhares de pontos de colocação sem rótulo, onde o PINO é punido matematicamente se violar as equações diferenciais de Edwards e Debye-Hückel. Se a física vacilar e a incerteza subir, um ciclo de Active Learning chama o SCFT sob demanda para corrigir a falha.
6. **Precisão Absoluta no Front-end**: Como o PINO prevê o resultado em 0.05 segundos com extrema precisão, sua inferência servirá como o chute inicial para um solucionador de passo único (Single-Step Newton-Raphson), garantindo conservação de massa e eletroneutralidade prefeita no WebApp.

---

## Roadmap e Status de Execução (To-Do List)

Abaixo estão os passos práticos detalhados na nossa "Pipeline PINO V3", a serem executados gradativamente:

- [x] **Fase 1: Amostragem e Geração de Espaço Contínuo**
  - Implementado o script de `Gaussian Random Fields` cujos parâmetros estruturais e físicos foram sorteados rigorosamente via **Latin Hypercube Sampling (LHS)**. O dataset original foi expandido offline para 40.000 amostras via Data Augmentation (simetria D4).
- [x] **Fase 2: Arquitetura Base da Rede Neural (PINO)**
  - Implementada a arquitetura expandida multicanal (Input de 8 camadas) para absorver o GRF, Debye e as malhas numéricas absolutas ($x, z$).
- [x] **Fase 2: Estruturação da Loss de Sobolev**
  - Esqueleto inicial de retropropagação e extração do Jacobiano configurado usando `torch.autograd`.
- [x] **Fase 1: Warm-Up Ground Truth (SCFT Real)**
  - Rodar o Solver tradicional de SCFT nos 5.000 Campos GRFs gerados para extrair a Densidade Polimérica exata $\phi_{scft}$.
- [ ] **Fase 2: Treinamento Base (Warm-Up Model)**
  - Treinar o PINO nas 5.000 amostras ancoradas focando no Erro Quadrático Médio e na Função de Perda de Sobolev por 1.000 épocas.
- [ ] **Fase 2: Aprendizado da Física (Collocation Points)**
  - Gerar 50.000 cenários físicos "vazios" (sem resolução no SCFT) e injetar as Equações Diferenciais na Loss para treinar a aderência às leis da termodinâmica.
- [ ] **Fase 3: Active Learning e Amostragem Adaptativa**
  - Criar um loop que inspeciona a variância e a falha das predições, rodando o SCFT de volta para geometrias de falha sistêmica.
- [ ] **Fase 3: O Filtro Físico de Newton-Raphson (Pós-Inferência)**
  - Construir o acoplamento do modelo à correção analítica clássica em tempo real para precisão absoluta.
- [ ] **Fase 4: Deploy no WebApp V3**
  - Desenvolver a interface final com as novas predições ativas, controle de sequências de carga via Fourier e campos paramétricos customizados.
