# ProjectLog

Diário do projeto. Entradas mais recentes primeiro. Horários em UTC.

## 2026-09-30 12:20 — Revisão do usuário: numeração, hidrogênios, visualização do sistema

**Pedidos:** (1) numerar as células; (2) "na preparação o n de átomos não muda — os H foram
realmente adicionados?"; (3) na visualização do sistema solvatado a água não aparece e a
proteína não está no centro da caixa.

**Diagnóstico (com evidência, não suposição)**
- (2) 1L2Y é RMN e já traz 150 H. `PDBFixer.addMissingHydrogens` → `Modeller.addHydrogens`
  mantém H existentes e só acrescenta os que faltam; testei removendo os H (304 → 154 átomos)
  e o PDBFixer recolocou exatamente 150 H com os mesmos nomes (304). Ou seja, o passo
  funcionava, mas não era demonstrável no caso RMN.
- (3) Reproduzi a visualização com Chromium headless (Playwright + 3Dmol.js 2.5.5 inline) e
  interroguei o viewer por JavaScript: o parser PDB do 3Dmol.js **descarta hidrogênios por
  padrão** (2459 átomos carregados de 7193); cada água virava um O isolado sem ligações, e o
  estilo `line` só desenha ligações → água invisível. Descentragem: `Modeller.addSolvent`
  centra a água no soluto (que estava na origem), enquanto a célula unitária do OpenMM (e o
  `addUnitCell` do 3Dmol) vai de (0,0,0) aos vetores a,b,c → proteína no canto.

**Correções**
- (1) Pós-processamento no gerador: toda célula recebe `[Célula N]` (markdown: `<sub>`, código:
  comentário na 1ª linha). 39 células numeradas.
- (2) Novo parâmetro `STRIP_HYDROGENS = True` (Seção 0). A Célula 16 remove os H do modelo de
  RMN (simulando o caso cristalográfico), imprime a composição por elemento antes/depois
  (`{'C': 98, 'N': 27, 'O': 29}` → `{'C': 98, 'H': 150, 'N': 27, 'O': 29}`), o número de H
  adicionados (150) e os H ligados ao N-terminal (`H, H2, H3` = NH₃⁺ em pH 7). Texto da Seção 4
  explica por que a contagem não mudava antes.
- (3a) `show_structure` passa `{"keepH": True}` a `addModel`, com comentário explicando o
  comportamento do 3Dmol.js. Água agora aparece como linhas O–H; os H da proteína aparecem como
  bastões (útil também para a Célula 17, que mostra os H recém-adicionados).
- (3b) Célula 19 translada o sistema para que o centro da caixa envolvente da proteína coincida
  com o centro da célula periódica (deslocamento impresso, ~2,1–2,2 nm por eixo). Efeito:
  centro da proteína em (22,5; 19,5; 20,9) Å numa caixa de 42,6 Å, inclusive após
  `enforcePeriodicBox` na minimização.
- Renderizações de verificação (headless): água + caixa + proteína centrada OK; hidrogênios
  visíveis na proteína preparada OK.

**Re-validação:** notebook completo re-executado em CPU com `QUICK_TEST=True`, 23/23 células
sem erro; 7232 átomos (2305 águas, 6 Na⁺, 7 Cl⁻), carga 0, densidade NPT → ~1,00 g/mL.

## 2026-09-29 19:27 — Validação local (CPU, QUICK_TEST) e correções

**Execução ponta a ponta** do notebook com `QUICK_TEST=True` (4 ps NVT + 10 ps NPT) via
`jupyter nbconvert --execute` no venv local (OpenMM 8.6.1, plataforma CPU). Resultado: 23/23
células de código sem erro. Números da execução (só para conferir sanidade, não são resultados
científicos):

| Item | Valor |
|---|---|
| Átomos (proteína / água / íons / total) | 304 / 2292 HOH / 6 Na⁺ 7 Cl⁻ / 7193 |
| Caixa cúbica | 4,25 nm de aresta (padding 1,5 nm) |
| Carga total após neutralização | +0,0000 e |
| Massa de H após HMR | 4,000 u |
| Energia potencial antes → depois da minimização | −35 323 → −121 022 kJ/mol |
| Densidade NPT (n = 50 amostras em 10 ps) | 0,990 ± 0,017 g/mL (ainda subindo: 10 ps é curto) |
| Temperatura NPT (n = 50) | 299,7 ± 5,6 K |
| RMSD backbone vs. minimizada (n = 10 frames) | média 0,75 Å, máx 0,94 Å |
| Velocidade em CPU (2 núcleos, sem GPU) | ~16–18 ns/dia |

**Bugs encontrados e corrigidos nesta validação**
1. Ao continuar a NPT com `Context.setState(estado_NVT)`, o OpenMM 8.x também restaura o tempo
   e o contador de passos → barra de progresso passava de 100 % e o eixo de tempo ficava
   deslocado duas vezes. Correção: `setTime(0)` e `setStepCount(0)` após `setState`, com
   comentário explicando.
2. Semântica do `padding` em `Modeller.addSolvent`: aresta = diâmetro + padding (lido no
   código-fonte, `width = max(2*radius+padding, 2*padding)`), ou seja, padding é a distância
   mínima proteína–imagem, e não proteína–borda como no GROMACS. O texto original dizia o
   contrário. Corrigido o texto e aumentado o padding de 1,0 para 1,5 nm para que a distância
   proteína–imagem fique acima do raio de corte (1,0 nm).
3. Avisos cosméticos do MDAnalysis ao gravar PDB (cargas formais, CRYST1) silenciados com
   filtro específico e comentário.
4. Versão do PDBFixer lida via `importlib.metadata` (o pacote não expõe `__version__`).

**Observação registrada no notebook:** nos primeiros ps da NVT a temperatura cai (~170 K na
primeira amostra) porque metade da energia cinética sorteada vira potencial (equipartição
após minimização); o termostato recupera em poucos ps. Esperado, não é erro.

**Arquivos no repositório:** `openmm_1L2Y_colab.ipynb` (sem saídas, 39 células),
`tools/build_notebook.py` (gerador; `python tools/build_notebook.py openmm_1L2Y_colab.ipynb`
reconstrói o notebook), `README.md`, `ProjectLog.md`, `.gitignore`.

## 2026-09-29 19:20 — Notebook Colab de MD da Trp-cage (1L2Y) criado

**Pedido:** notebook `.ipynb` para Google Colab (GPU) que baixe 1L2Y, mostre a proteína, prepare
(PDBFixer), monte o sistema (solvente explícito, all-atom), mostre de novo, minimize, rode NVT
(1 ns) e NPT (5 ns), com uma célula listando os campos de força suportados pelo OpenMM. Sem GUI.
Tudo claramente comentado.

**O que foi feito**
- `openmm_1L2Y_colab.ipynb` (39 células; gerado por script `nbformat` para evitar erros de JSON).
- `README.md` com roadmap em checkboxes; `.gitignore` para saídas de simulação.

**Decisões e justificativas**
- *Campo de força:* Amber ff14SB (`amber14/protein.ff14SB.xml`, Maier et al. 2015) + TIP3P com
  íons (`amber14/tip3p.xml`). É a combinação marcada como "recomendada" no manual do OpenMM 8.6
  (seção 3.6 do guia, tabela Amber14) e a mais testada da família. ff19SB+OPC documentado como
  alternativa (duas strings na Seção 0).
- *Caixa:* cúbica, padding 1,0 nm (garante > 2× o raio de corte entre imagens periódicas),
  0,15 M NaCl + neutralização. Forma cúbica escolhida pela simplicidade didática; dodecaedro é
  opção documentada (`BOX_SHAPE`).
- *Não ligadas:* PME, corte 1,0 nm, tolerância 5e-4 (padrões do OpenMM).
- *Integrador:* `LangevinMiddleIntegrator` 300 K, fricção 1/ps (esquema de Zhang et al. 2019,
  usado nos exemplos oficiais do OpenMM).
- *Passo de tempo:* 4 fs com HMR (`hydrogenMass=4 amu`, valor do exemplo da seção 3.7.7 do
  manual) + `HBonds` + água rígida, conforme Hopkins et al. 2015. `USE_HMR=False` cai para 2 fs.
- *Protocolo:* minimização (L-BFGS, tolerância 10 kJ/mol/nm) → NVT 1 ns com restrições
  harmônicas (k = 1000 kJ/mol/nm²) nos átomos pesados da proteína (relaxar solvente sem
  perturbar a estrutura de RMN) → NPT 5 ns sem restrições com `MonteCarloBarostat` 1 bar.
  Cada fase usa uma `Simulation` própria: a de NPT recebe o estado completo (posições,
  velocidades, caixa) da NVT via `Context.setState`. Motivo: remover a força de restrição e
  adicionar o barostato altera o `System`, o que exige um novo `Context`; fazer isso
  explicitamente é mais claro do que truques com parâmetros globais.
- *Saídas:* XTC (compacto, lido pelo MDAnalysis) a cada 10 ps; CSV de observáveis a cada 1 ps;
  checkpoint a cada 100 ps; PDB e XML de estado no fim de cada fase.
- *Estrutura:* 1L2Y é ensemble de RMN (38 modelos); o notebook extrai explicitamente o modelo 1
  antes do PDBFixer, para que todas as etapas partam da mesma coordenada.
- *Instalação:* `pip install openmm[cuda12] pdbfixer MDAnalysis py3Dmol`. O extra `cuda12`
  instala o plugin `OpenMM-CUDA-12` (verificado nos metadados do wheel 8.6.1). A célula só
  instala o que falta e testa de fato cada plataforma (CUDA → OpenCL → CPU) criando um
  `Context`, em vez de confiar apenas na lista de plataformas compiladas.
- *Visualização:* py3Dmol (funciona no Colab sem widgets). Animação usa frames alinhados pelo
  MDAnalysis, só a proteína, ≤ ~50 frames.
- *Análises:* RMSD do backbone vs. estrutura minimizada, RMSF de Cα (procedimento
  alinhar→média→realinhar do MDAnalysis), raio de giro, T/densidade/E/V das duas fases. Cada
  gráfico com um único eixo y. Números impressos sempre acompanhados de n (frames/amostras).
- *Integridade:* a Seção 10 do notebook afirma explicitamente que 1 trajetória de 5 ns não
  permite conclusões sobre estabilidade/termodinâmica da Trp-cage (tempos de enovelamento na
  escala de µs) e que frames consecutivos são correlacionados.

**Verificações**
- Todos os 18 DOIs citados foram resolvidos no registro de handles do doi.org (HTTP 200) em
  2026-09-29 antes de entrarem no notebook.
- Trechos relevantes do manual do OpenMM 8.6 (instalação via pip com extra `cuda12`, tabela
  de campos de força, `hydrogenMass`, `addSolvent`, barostato) lidos em docs.openmm.org.
- Ambiente local (sem GPU): OpenMM 8.6.1, PDBFixer 1.12.0, MDAnalysis 2.10.0. O `pip` do
  sistema (Debian) falhou ao construir o wheel do pdbfixer (`install_layout`, bug do
  setuptools empacotado pelo Debian); em um venv com setuptools atualizado a instalação
  funcionou. Isso não afeta o Colab, mas está registrado.
- Execução ponta a ponta do notebook com `QUICK_TEST=True` em CPU: ver entrada seguinte.

**Pendente**
- Validar no Colab com GPU e os parâmetros completos (1 + 5 ns) e registrar ns/dia na T4.
  Não há GPU neste ambiente; não vou inventar números de desempenho.
