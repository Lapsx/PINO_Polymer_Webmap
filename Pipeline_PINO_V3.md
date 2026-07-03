# 🚀 Arquitetura e Pipeline de Treinamento - V3 (PINO Polieletrólito)

A versão V3 não será apenas um avanço em relação aos dados; será uma mudança de paradigma. Sairemos de um mapeamento puramente estatístico (Supervisionado/FNO) para um mapeador guiado por leis fundamentais (Física-Informada/PINO).

Aqui está o pipeline completo de ponta a ponta estruturado com base nas ideias documentadas.

---

## FASE 1: Engenharia do Dataset e Amostragem (Otimizando o Espaço Infinito)

O gargalo de combinações da distribuição de carga $\sigma(s)$ e dos campos espaciais $\omega(\mathbf{r})$ impede a força bruta. O pipeline de amostragem será feito em 3 camadas:

### 1.1 Representação de Sequência via Séries de Fourier
O polímero não será representado como um vetor discreto $[+1, -1, 0, \dots]$. Transformaremos a densidade de carga da cadeia $\sigma(s)$ em seus primeiros $K$ coeficientes de Fourier (ex: $K=8$). A rede aprenderá sobre as "frequências" de carga da cadeia (ex: blocos longos vs. alternância rápida), reduzindo o espaço de amostragem de $3^{100}$ para apenas 8 dimensões contínuas.

### 1.2 Campos Espaciais Aleatórios Gaussiano (GRFs)
Para simular as paredes, cargas externas e potenciais, o gerador de campos não usará formatos rígidos. O campo de energia basal será gerado via **Gaussian Random Fields**. Ao ajustar o *correlation length* (comprimento de correlação) do GRF, podemos simular desde superfícies contínuas até agrupamentos nano-estericos irregulares.

### 1.3 Geração Híbrida de Dados (Supervisionado + Não-Supervisionado)
O solver clássico de SCFT será usado com parcimônia para evitar semanas de simulação.
* **Dados Ancorados (Ground Truth):** Somente $\sim 5.000$ simulações completas do SCFT serão resolvidas, com foco em regimes de transição de fase e sequências básicas (Diblock, Triblock, Homopolímero).
* **Dados Não-Rótulados (PINO Collocation Points):** Serão gerados $\sim 50.000$ cenários (compostos por diferentes GRFs e sequências de Fourier) onde o SCFT **não** será rodado. Apenas o espaço e os parâmetros serão entregues para a rede aprender pela Função de Perda Física (Residuals).

---

## FASE 2: Modificações Arquiteturais (A Transição para PINO)

O coração do V3 é o *Physics-Informed Neural Operator*. A rede recebe entradas físicas e sua saída deve respeitar leis termodinâmicas no próprio *backpropagation*.

### 2.1 A Nova Assinatura da Rede (Input/Output)
A entrada da FNO será um tensor multicanal em $\mathbf{r}$:
`[ Campo Externo ω(r), Potencial Debye ψ(r), Máscara de Parede H(r) ]` + um vetor global de parâmetros `[ b, u, κ, C_fourier_1, ..., C_fourier_k ]`
**Saída:** `Densidade do Polímero ϕ(r)`

### 2.2 O Treinamento de Sobolev (O Poder das Derivadas)
Essa é a estratégia mais matadora. A Função de Perda (Loss) não medirá apenas a diferença entre a densidade prevista e a real. Ela usará as derivadas!
$$ \mathcal{L}_{Total} = \lambda_1 || \phi_{pred} - \phi_{scft} ||^2 + \lambda_2 \left|\left| \frac{\delta \phi_{pred}}{\delta \omega} - \frac{\delta \phi_{scft}}{\delta \omega} \right|\right|^2 + \lambda_3 ||\mathcal{F}_{Fisica}||^2 $$
O termo $\lambda_2$ (Sobolev Loss) força a rede a aprender a "forma" da função e as suscetibilidades térmicas da molécula. Matematicamente, isso equivale a fornecer os tensores de correlação RPA para a rede, multiplicando a eficiência dos dados ancorados por 10x.

### 2.3 Resíduos Físicos (O Coração do PINO)
O termo $\lambda_3$ acima é onde a física entra sem precisar de dados. Para os 50.000 pontos não-rotulados, passamos a previsão $\phi_{pred}$ através das Equações Diferenciais de Edwards e Debye-Hückel via *Autograd*. Se a rede previr algo que não zera as equações diferenciais, ela é penalizada. 

---

## FASE 3: Correção Pós-Inferência em Tempo Real (O "Filtro" Físico)

Uma vez que o modelo esteja treinado e em produção no WebApp, a inferência ultrarrápida da rede passará por um último passo de "Limpeza Termodinâmica".

### 3.1 Newton-Raphson Single-Step
Como a nossa rede (já provado no WebApp V1.0) consegue expelir o Jacobiano analítico da Densidade pelo Potencial $\left( \frac{\delta \phi}{\delta \omega} \right)$ de forma impecável via Autograd, podemos acoplar a rede a um corretor clássico.
1. O PINO chuta a densidade em $0.05$ segundos com $98\%$ de precisão.
2. Usamos o Jacobiano extraído instantaneamente para dar apenas $1$ único passo de Newton-Raphson.
3. Resultado: A densidade converge para a precisão exata da máquina (conservação de massa perfeita) em fração de segundo.

### 3.2 Eletro-Neutralidade Rigorosa
Uma camada de projeção simples (Filtro Linear) será aplicada à saída Newton-Raphson garantindo que a carga total integrada da solução mais o polímero zere com as paredes carregadas (Condição de Eletroneutralidade Global).

---

## Ciclo de Vida Prático (Pipeline de Execução)

1. **Warm-Up:** SCFT (Solver Tradicional) gera 5.000 amostras em 2 dias rodando GRFs.
2. **Treino Base:** Treinamos o PINO apenas com `Loss de Sobolev` por 1.000 épocas.
3. **Aprendizado da Física:** Injetamos os $50.000$ pontos de colocação e ativamos a `Loss Física`. A rede treina por mais 2.000 épocas aprendendo a "respeitar as equações diferenciais".
4. **Active Learning (Ciclo Fechado):** Medimos a variância espacial das previsões. Nas 50 geometrias onde o PINO falhar vergonhosamente, rodamos o SCFT de novo para aqueles exatos casos e adicionamos na base ancorada.
5. **Deploy:** Integramos ao novo WebApp com o Newton-Raphson dinâmico.
