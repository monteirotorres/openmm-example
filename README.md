# openmm-example

Exemplo didático de dinâmica molecular (MD) *all-atom* em solvente explícito da miniproteína
Trp-cage (PDB **1L2Y**) com **OpenMM** + **PDBFixer** + **MDAnalysis**, feito para rodar na GPU
do **Google Colab** sem interface gráfica.

## Notebook

`openmm_1L2Y_colab.ipynb`

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monteirotorres/openmm-example/blob/main/openmm_1L2Y_colab.ipynb)

Pipeline: download do PDB → visualização (py3Dmol) → preparo (PDBFixer, pH 7) → solvatação e
íons (Amber ff14SB + TIP3P, 0,15 M NaCl) → visualização → minimização → NVT 1 ns (proteína
restrita) → NPT 5 ns (produção) → análise (RMSD, RMSF, raio de giro, T, densidade) e animação.
Inclui uma célula que lista todos os campos de força XML incluídos na versão instalada do OpenMM.

No Colab: *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)* e depois
*Executar tudo*. Todos os parâmetros estão na Seção 0; `QUICK_TEST = True` roda o pipeline
inteiro em poucos segundos de MD para validação.

## Roadmap

- [x] Definir protocolo e verificar referências (DOIs resolvidos via doi.org)
- [x] Notebook Colab
  - [x] Parâmetros centralizados (Seção 0) e modo `QUICK_TEST`
  - [x] Detecção de GPU, instalação (`openmm[cuda12]`, pdbfixer, MDAnalysis, py3Dmol)
  - [x] Escolha de plataforma (CUDA → OpenCL → CPU) e `openmm.testInstallation`
  - [x] Célula listando os campos de força incluídos no OpenMM (dinâmica) + guia em tabela
  - [x] Download de 1L2Y, extração do modelo 1, visualização
  - [x] Preparo com PDBFixer com relatório de cada etapa
  - [x] Montagem do sistema (ff14SB/TIP3P, caixa cúbica, 0,15 M NaCl), verificação de carga
  - [x] Minimização com energia antes/depois e deslocamento RMS
  - [x] NVT 1 ns com restrições harmônicas nos átomos pesados
  - [x] NPT 5 ns com barostato de Monte Carlo
  - [x] Análise com MDAnalysis (RMSD, RMSF, Rg) e observáveis (T, densidade, E, V)
  - [x] Animação da trajetória e sobreposição início/fim
  - [x] Seção sobre limites de interpretação e referências
- [x] Validação local do pipeline completo em CPU (`QUICK_TEST = True`)
- [ ] Validação em GPU no Colab com os parâmetros completos (1 ns NVT + 5 ns NPT)
  - [ ] Registrar velocidade (ns/dia) obtida na T4 e tempo total
  - [ ] Confirmar que `openmm[cuda12]` carrega a plataforma CUDA no Colab atual
- [ ] Extensões possíveis (não pedidas): replicatas, ff19SB/OPC, análise de contatos nativos

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `openmm_1L2Y_colab.ipynb` | O notebook |
| `README.md` | Este arquivo (roadmap) |
| `ProjectLog.md` | Diário do projeto: o que foi feito, por quê, quando (mais recente primeiro) |
| `tools/build_notebook.py` | Gerador do notebook (`python tools/build_notebook.py openmm_1L2Y_colab.ipynb`); edite aqui e reconstrua para manter diffs legíveis |
