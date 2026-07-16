# PINO Polymer Webmap (V3)

## Objetivo do Projeto

O **PINO Polymer Webmap** é a versão mais recente (V3) da arquitetura de predição de densidades poliméricas. Nascido como um Fourier Neural Operator (FNO) clássico em versões anteriores, o projeto tem como objetivo principal alterar **o paradigma de mapeamento puramente estatístico (dados para dados) usado anteriormente, em direção a um aprendizado profundo guiado pelas leis fundamentais da mecanica estatistica**.

A base teórica e de engenharia desta versão introduz o **Physics-Informed Neural Operator (PINO)**, buscando criar um modelo capaz de generalizar cenários de confinamento e interação eletrostática sobre polieletrólitos sem depender exclusivamente de simulações exaustivas de Self-Consistent Field Theory (SCFT).

Os principais pontos teóricos que sustentam este objetivo são:
1. **Representação Contínua da Carga**: Utilização de coeficientes de Fourier para modelar sequências de carga do polímero, comprimindo um espaço combinatório muito grande para um espaço de frequência reduzido e interpretável.
2. **Diversidade Topológica Infinita**: Substituição de cenários rígidos de ancoragem por *Gaussian Random Fields (GRFs)* com parâmetros para simular as mais variadas topologias de confinamento e potenciais. O espaço paramétrico que dita a topologia desses campos e as propriedades termodinâmicas do polímero são amostrados usando **Latin Hypercube Sampling (LHS)**, garantindo uma boa cobertura estatística.
3. **Exploração da Simetria Física (Data Augmentation D4)**: Pelo fato da física de matrizes de confinamento e a resposta polimérica ($\phi$) serem espacialmente isotrópicas e invariantes a rotações e espelhamentos, aplica-se o Grupo de Simetria D4 (rotações 90º, 180º, 270º e reflexões) offline para multiplicar exponencialmente o *Ground Truth* do SCFT sem custo computacional, mitigando o *overfitting*, aumentando o espaço de fase do treino e evitando respostas paradoxais como artefatos do treino.
4. **Treinamento de Sobolev**: Injeção das derivadas do modelo na Função de Custo (*Loss*). A rede não aprende apenas a densidade $\phi(\mathbf{r})$, mas também a sua suscetibilidade térmica exata $\left( \frac{\delta \phi}{\delta V} \right)$, aumentando o valor informacional de cada inferencia.
5. **Resíduos Físicos & Aprendizado Ativo**: Utilização de dezenas de milhares de pontos de colocação sem rótulo, onde o PINO é punido matematicamente se violar as equações diferenciais de Edwards e Debye-Hückel. Se a física falhar e a incerteza subir, um ciclo de Active Learning chama o SCFT sob demanda para corrigir a falha.
6. **Precisão Ampliada no Front-end**: Como o PINO prevê o resultado em 0.05 segundos com extrema precisão, sua inferência servirá como o chute inicial para um solucionador de passo único (Single-Step Newton-Raphson), garantindo conservação de massa e eletroneutralidade no WebApp.

---

## Roadmap e Status de Execução (To-Do List)

Abaixo estão os passos práticos detalhados na nossa "Pipeline PINO V3", a serem executados gradativamente:

- [x] **Fase 1: Amostragem e Geração de Espaço Contínuo**
  - Implementado o script de `Gaussian Random Fields` cujos parâmetros estruturais e físicos foram sorteados rigorosamente via **Latin Hypercube Sampling (LHS)**. O dataset original foi expandido offline para 40.000 amostras via Data Augmentation (simetria D4).
- [x] **Fase 2: Arquitetura Base da Rede Neural (PINO)**
  - Implementada a arquitetura expandida multicanal (Input de 8 camadas) para absorver o GRF, Debye e as malhas numéricas absolutas ($x, z$).
- [x] **Fase 3: Estruturação da Loss de Sobolev**
  - Esqueleto inicial de retropropagação e extração do Jacobiano configurado usando `torch.autograd`.
- [x] **Fase 4: Warm-Up Ground Truth (SCFT Real)**
  - Rodar o Solver tradicional de SCFT nos 5.000 Campos GRFs gerados para extrair a Densidade Polimérica exata $\phi_{scft}$.
- [x] **Fase 5: Treinamento Base (Warm-Up Model)**
  - Treinar o PINO nas 5.000 amostras ancoradas focando no Erro Quadrático Médio e na Função de Perda de Sobolev por 1.000 épocas.
- [x] **Fase 6: Deploy no WebApp V3**
  - Desenvolver a interface final com as novas predições ativas, controle de sequências de carga via Fourier e campos paramétricos customizados.
- [x] **Fase 7/Extensão: Treinamento de Cargas de Polímero (Diblock e Alternado)**
  - Criação de proxies físicos (perturbações) para simular o comportamento de polímeros Diblock (C1) e Alternados (C4), e realizar o fine-tuning da rede (25 épocas).
- [x] **Fase 8/Extensão: Slider de Intensidade de Carga do Polímero**
  - Refatorar o Frontend e o Dataset para treinar a PINO em valores contínuos de intensidade de carga (ex: 0.0 a 2.0), em vez de apenas um seletor booleano.
- [X] **Fase 9: O Filtro Físico de Newton-Raphson (Pós-Inferência)**
  - Construir o acoplamento do modelo à correção analítica clássica em tempo real para melhor precisão.
- [/] **Fase 10: Aprendizado da Física (Collocation Points)**
  - Gerar 50.000 cenários físicos "vazios" (sem resolução no SCFT) e injetar as Equações Diferenciais na Loss para treinar a aderência à termodinâmica do sistema.
- [ ] **Fase 11: Active Learning e Amostragem Adaptativa**
  - Criar um loop que inspeciona a variância e a falha das predições, rodando o SCFT de volta para geometrias de falha sistêmica.
- [ ] **Fase 12: High-Throughput Screening (HTS) no WebApp**
  - Adicionar um botão de upload de CSV/JSON no WebApp para computar milhares de configurações num único Batch Forward Pass da PINO, retornando relatórios estatísticos da predição instantaneamente.
