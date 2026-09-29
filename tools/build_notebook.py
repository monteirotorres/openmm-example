#!/usr/bin/env python
"""Gera o notebook `openmm_1L2Y_colab.ipynb` a partir de células definidas aqui.

Manter o notebook como código Python facilita revisão em git (diffs legíveis)
e evita erros de sintaxe JSON. Rode:  python build_notebook.py <saida.ipynb>
"""
import sys
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Dinâmica molecular da Trp-cage (PDB 1L2Y) com OpenMM + MDAnalysis no Google Colab

Exemplo **simples, completo e comentado** de uma simulação de dinâmica molecular (MD)
*all-atom* em **solvente explícito**, rodando na **GPU do Google Colab**, sem interface gráfica
(sem OpenMM-Setup / CHARMM-GUI).

**Sistema:** Trp-cage TC5b (PDB [1L2Y](https://www.rcsb.org/structure/1L2Y)), uma miniproteína
de 20 resíduos (sequência `NLYIQWLKDGGPSSGRPPPS`) desenhada por Neidigh, Fesinmeyer & Andersen
(2002) [1]. É um dos menores sistemas com enovelamento cooperativo e, por isso, um
sistema-teste clássico em MD. A estrutura é um conjunto (*ensemble*) de RMN com 38 modelos;
usamos o modelo 1.

**Pipeline (uma seção por etapa):**

| # | Etapa | Ferramenta |
|---|-------|-----------|
| 0 | Parâmetros da simulação (tudo em um só lugar) | Python |
| 1 | Verificar GPU e instalar dependências | `pip` |
| 2 | Listar plataformas de cálculo e **campos de força (FF) disponíveis no OpenMM** | OpenMM |
| 3 | Baixar 1L2Y do RCSB e **visualizar** | `urllib`, py3Dmol |
| 4 | **Preparar** a proteína (resíduos/átomos faltantes, hidrogênios em pH 7) | PDBFixer |
| 5 | **Montar o sistema**: FF, caixa de água, íons; **visualizar** | OpenMM `Modeller` |
| 6 | **Minimizar** a energia | OpenMM |
| 7 | **NVT** (1 ns): equilibrar a temperatura com a proteína restrita | OpenMM |
| 8 | **NPT** (5 ns): produção a 300 K e 1 bar | OpenMM |
| 9 | **Análise**: RMSD, RMSF, raio de giro, observáveis termodinâmicos; animação | MDAnalysis, py3Dmol |

**Antes de rodar:** em *Ambiente de execução → Alterar o tipo de ambiente de execução*,
escolha um acelerador de hardware **GPU** (T4 é suficiente). Sem GPU o notebook ainda
funciona (cai para OpenCL ou CPU), mas 6 ns de MD podem levar horas.

**Como usar:** execute as células em ordem (*Ambiente de execução → Executar tudo*).
Todos os parâmetros ajustáveis estão na Seção 0. Para um teste rápido de que o pipeline
inteiro funciona (segundos de MD em vez de nanossegundos), ponha `QUICK_TEST = True`.

> **Aviso científico.** Este é um exemplo didático: **uma** trajetória de 5 ns de produção. Isso
> é suficiente para ilustrar o protocolo e as análises, mas **não** é suficiente para tirar
> conclusões sobre estabilidade, enovelamento ou termodinâmica da Trp-cage, cujos tempos de
> (des)enovelamento são da ordem de microssegundos [1]. Veja a Seção 10.

Referências numeradas estão na última célula. Todos os DOIs foram verificados.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Parâmetros da simulação

Tudo o que você pode querer mudar está aqui. As escolhas padrão e a justificativa de cada uma:

- **Campo de força:** Amber **ff14SB** para a proteína [2] com água **TIP3P** [3] e íons
  compatíveis (`amber14/tip3p.xml`). É a combinação recomendada pela documentação do OpenMM
  para proteínas e a mais amplamente testada da família Amber. Alternativa mais recente:
  `amber19/protein.ff19SB.xml` + `amber19/opc.xml` (ff19SB [4] foi parametrizado junto com a
  água OPC [5]); basta trocar as duas strings abaixo.
- **Caixa:** cúbica, com *padding* de 1,5 nm. Atenção à convenção do OpenMM: a aresta da caixa
  é (diâmetro da esfera envolvente da proteína) + *padding*, ou seja, o *padding* é a **distância
  mínima entre a proteína e a sua imagem periódica** (diferente do `-d` do GROMACS, que é a
  distância até a borda). Com 1,5 nm essa distância fica acima do raio de corte de 1,0 nm.
  NaCl a 0,15 M (fisiológico) + íons de neutralização.
- **Eletrostática:** PME (*Particle Mesh Ewald*) [6], raio de corte 1,0 nm. Correção de
  dispersão de longo alcance ligada (padrão do OpenMM).
- **Integrador:** `LangevinMiddleIntegrator` (esquema *middle* de Zhang et al. [7], termostato
  de Langevin com fricção 1 ps⁻¹) a 300 K.
- **Passo de tempo:** 4 fs, possível porque (i) ligações com hidrogênio são mantidas rígidas
  (`HBonds`), (ii) a água é rígida e (iii) usamos **HMR** (*hydrogen mass repartitioning*:
  a massa dos H é elevada para 4 u e a diferença é subtraída do átomo pesado ligado, mantendo a
  massa total) [8]. Sem HMR use 2 fs.
- **Barostato:** Monte Carlo isotrópico [9, 10] a 1 bar, tentativa a cada 25 passos (padrão).
- **Restrição na fase NVT:** restrições harmônicas nos átomos pesados da proteína
  (k = 1000 kJ mol⁻¹ nm⁻²) para relaxar o solvente sem perturbar a estrutura experimental.
  Removidas na fase NPT.
""")

code(r"""
# =============================================================================
# Seção 0 — Parâmetros (edite aqui; o resto do notebook lê destas variáveis)
# =============================================================================

# ----- Sistema ---------------------------------------------------------------
PDB_ID = "1L2Y"          # Trp-cage TC5b, ensemble de RMN com 38 modelos
NMR_MODEL = 0            # índice (base 0) do modelo do ensemble a usar (0 = modelo 1)
PH = 7.0                 # pH usado pelo PDBFixer para escolher estados de protonação

# ----- Campo de força (arquivos XML incluídos no OpenMM; veja a lista na Seção 2) ---
FF_PROTEIN = "amber14/protein.ff14SB.xml"   # Amber ff14SB (Maier et al. 2015)
FF_SOLVENT = "amber14/tip3p.xml"            # água TIP3P + parâmetros de íons compatíveis
WATER_MODEL = "tip3p"    # modelo geométrico de água que o Modeller vai inserir
                         # (precisa ser coerente com FF_SOLVENT: tip3p/spce/tip4pew/tip5p)

# ----- Caixa e solvente ------------------------------------------------------
BOX_PADDING_NM = 1.5     # aresta = diâmetro da proteína + padding → distância mínima
                         # entre a proteína e sua imagem periódica (nm); manter > corte
BOX_SHAPE = "cube"       # 'cube' | 'dodecahedron' | 'octahedron' (os dois últimos são menores)
IONIC_STRENGTH_M = 0.15  # força iônica adicional (mol/L) além dos íons neutralizantes
POSITIVE_ION, NEGATIVE_ION = "Na+", "Cl-"

# ----- Interações não ligadas -----------------------------------------------
NONBONDED_CUTOFF_NM = 1.0   # raio de corte real (nm) para LJ e parte real do PME
EWALD_ERROR_TOL = 5e-4      # tolerância do PME (padrão do OpenMM)

# ----- Integração e termostato ----------------------------------------------
TEMPERATURE_K = 300.0    # temperatura alvo (K)
PRESSURE_BAR = 1.0       # pressão alvo da fase NPT (bar)
FRICTION_PER_PS = 1.0    # coeficiente de fricção do termostato de Langevin (1/ps)
USE_HMR = True           # hydrogen mass repartitioning → permite 4 fs
HYDROGEN_MASS_AMU = 4.0  # massa atribuída aos H (u) quando USE_HMR=True
TIMESTEP_FS = 4.0 if USE_HMR else 2.0
RANDOM_SEED = 2026       # semente p/ velocidades iniciais, termostato e barostato

# ----- Restrições na fase NVT -----------------------------------------------
RESTRAINT_K = 1000.0     # kJ/(mol nm^2), aplicada aos átomos pesados da proteína

# ----- Duração e frequência de saída -------------------------------------------
QUICK_TEST = False       # True → segundos de MD, só para testar que tudo roda
NVT_NS = 1.0             # duração da equilibração NVT (ns)
NPT_NS = 5.0             # duração da produção NPT (ns)
TRAJ_INTERVAL_PS = 10.0  # intervalo entre frames gravados na trajetória (ps)
CSV_INTERVAL_PS = 1.0    # intervalo entre linhas dos observáveis (energia, T, densidade…)
STDOUT_INTERVAL_PS = 50.0  # intervalo dos relatórios de progresso impressos na tela
CHECKPOINT_INTERVAL_PS = 100.0  # intervalo de gravação de checkpoint (p/ retomar)
MINIMIZE_TOLERANCE = 10.0  # kJ/(mol nm): critério de parada da minimização

if QUICK_TEST:           # encurta tudo para validar o pipeline em ~1 minuto
    NVT_NS, NPT_NS = 0.004, 0.010
    TRAJ_INTERVAL_PS, CSV_INTERVAL_PS = 1.0, 0.2
    STDOUT_INTERVAL_PS, CHECKPOINT_INTERVAL_PS = 2.0, 5.0

# ----- Arquivos de saída ----------------------------------------------------
OUTPUT_DIR = "md_1L2Y"   # pasta onde tudo é gravado (relativa ao diretório atual)

print(f"Sistema {PDB_ID} | FF {FF_PROTEIN} + {FF_SOLVENT} | dt = {TIMESTEP_FS} fs "
      f"| NVT {NVT_NS} ns | NPT {NPT_NS} ns | QUICK_TEST={QUICK_TEST}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. GPU e instalação

O OpenMM é distribuído no PyPI. O pacote base traz as plataformas *Reference*, *CPU* e
*OpenCL*; o extra `openmm[cuda12]` adiciona a plataforma **CUDA** (a mais rápida em GPUs
NVIDIA, como a T4 do Colab). Também instalamos o **PDBFixer** (preparo de estruturas),
o **MDAnalysis** [11, 12] (análise de trajetórias) e o **py3Dmol** [13] (visualização 3D no
navegador). A instalação leva ~1–2 min.
""")

code(r"""
# =============================================================================
# Seção 1a — Há uma GPU disponível?
# =============================================================================
import subprocess, shutil

if shutil.which("nvidia-smi"):
    # nvidia-smi é o utilitário do driver NVIDIA; mostra modelo da GPU, memória e versão CUDA
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi não encontrado: não há GPU NVIDIA neste ambiente.\n"
          "No Colab: Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4).")
""")

code(r"""
# =============================================================================
# Seção 1b — Instalação das dependências (só instala o que estiver faltando)
# =============================================================================
import importlib, subprocess, sys

def _missing(module_name):
    # Retorna True se o módulo Python NÃO puder ser importado.
    try:
        importlib.import_module(module_name)
        return False
    except ImportError:
        return True

IN_COLAB = "google.colab" in sys.modules

# Pacote PyPI → nome do módulo Python correspondente
packages = {
    "openmm[cuda12]": "openmm",     # extra cuda12 = plataforma CUDA (ignorado sem GPU NVIDIA)
    "pdbfixer": "pdbfixer",
    "MDAnalysis": "MDAnalysis",
    "py3Dmol": "py3Dmol",
}
to_install = [pkg for pkg, mod in packages.items() if _missing(mod)]

if IN_COLAB and "openmm[cuda12]" not in to_install:
    # Mesmo que o openmm já exista, no Colab garantimos o plugin CUDA.
    to_install.append("openmm[cuda12]")

if to_install:
    print("Instalando:", " ".join(to_install))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    print("Instalação concluída.")
else:
    print("Todas as dependências já estão instaladas.")
""")

code(r"""
# =============================================================================
# Seção 1c — Importações e versões (registre-as ao relatar resultados)
# =============================================================================
import os, sys, time, math, textwrap, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmm
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3

import pdbfixer
import MDAnalysis as mda
import py3Dmol

# Pasta de saída: tudo o que a simulação grava vai para cá.
OUT = Path(OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)

print(f"Python     {sys.version.split()[0]}")
print(f"OpenMM     {openmm.__version__}")
from importlib.metadata import version as _pkg_version
print(f"PDBFixer   {_pkg_version('pdbfixer')}")
print(f"MDAnalysis {mda.__version__}")
print(f"py3Dmol    {py3Dmol.__version__}")
print(f"numpy      {np.__version__}")
print(f"Saídas em  {OUT.resolve()}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Plataformas de cálculo e campos de força disponíveis

### 2a. Plataformas
O OpenMM implementa as mesmas equações em várias *plataformas*: **CUDA** (GPU NVIDIA, a mais
rápida), **OpenCL** (GPUs de qualquer fabricante), **CPU** e **Reference** (implementação
lenta, apenas para validação). Escolhemos a melhor disponível. Em GPU usamos precisão `mixed`
(forças em precisão simples, acumulação/integração em dupla), o compromisso padrão entre
velocidade e conservação de energia.
""")

code(r"""
# =============================================================================
# Seção 2a — Plataformas disponíveis e escolha da melhor
# =============================================================================
available = [openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())]
print("Plataformas compiladas nesta instalação:", available)

# Ordem de preferência. Uma plataforma pode aparecer na lista e ainda assim falhar ao
# criar um Context (ex.: CUDA sem driver); por isso testamos de fato cada uma.
PLATFORM, PLATFORM_PROPERTIES = None, {}
for name, props in [("CUDA", {"Precision": "mixed"}),
                    ("OpenCL", {"Precision": "mixed"}),
                    ("CPU", {}),
                    ("Reference", {})]:
    if name not in available:
        continue
    try:
        # Teste mínimo: um sistema com 1 partícula, sem forças.
        _sys = openmm.System(); _sys.addParticle(1.0)
        _integ = openmm.VerletIntegrator(0.001)
        _plat = openmm.Platform.getPlatformByName(name)
        openmm.Context(_sys, _integ, _plat, props)
        PLATFORM, PLATFORM_PROPERTIES = _plat, props
        break
    except Exception as exc:
        print(f"  {name}: indisponível ({str(exc).splitlines()[0][:90]})")

if PLATFORM is None:
    raise RuntimeError("Nenhuma plataforma do OpenMM funcionou.")
print(f"\nPlataforma escolhida: {PLATFORM.getName()}  propriedades={PLATFORM_PROPERTIES}")
if PLATFORM.getName() in ("CPU", "Reference"):
    print("AVISO: sem GPU. 6 ns de MD podem levar muitas horas; considere QUICK_TEST=True.")

# Teste oficial de instalação: compara forças entre plataformas (diferenças ~1e-5 são normais).
print("\n--- python -m openmm.testInstallation ---")
print(subprocess.run([sys.executable, "-m", "openmm.testInstallation"],
                     capture_output=True, text=True).stdout)
""")

md(r"""
### 2b. Campos de força incluídos no OpenMM

O OpenMM descreve campos de força em arquivos **XML** carregados por `app.ForceField(...)`.
Um campo de força completo para um sistema solvatado é normalmente a combinação de **um arquivo
de biomoléculas** (proteína/DNA/RNA/lipídio) **+ um arquivo de água e íons**, ambos da mesma
família. A célula abaixo lista **todos os XML instalados na sua versão** (a lista é gerada
dinamicamente a partir do pacote, então reflete exatamente o que você tem). O guia abaixo
explica os principais.

| Família (arquivo) | O que contém | Observações |
|---|---|---|
| `amber19-all.xml` ou `amber19/protein.ff19SB.xml` … | Amber ff19SB (proteína) [4], DNA OL21, RNA OL3, lipid21 | Recomendado com água OPC (`amber19/opc.xml`) [5] |
| `amber14-all.xml` ou `amber14/protein.ff14SB.xml` … | Amber ff14SB (proteína) [2], DNA OL15, RNA OL3, lipid17, GLYCAM (carboidratos) | Água: `amber14/tip3p.xml`, `tip3pfb.xml`, `tip4pew.xml`, `opc.xml`, `spce.xml`… (cada um já traz íons) |
| `amber99sb.xml`, `amber99sbildn.xml`, `amber03.xml`, `amber96.xml`, `amber10.xml`, `amberfb15.xml` | Versões mais antigas da família Amber (só proteína/ácidos nucleicos) | Usar com `tip3p.xml` etc. (raiz) |
| `*_obc.xml` (ex. `amber99_obc.xml`) | Parâmetros de solvente implícito GB-OBC para o FF correspondente | Para o modelo implícito moderno use `implicit/*.xml` |
| `charmm36_2024.xml` (+ `charmm36_2024/water.xml`) | CHARMM36 (julho 2024) incl. CHARMM36m [14] para proteínas, lipídios, ácidos nucleicos, carboidratos | `charmm36.xml` é a versão anterior; não misture águas de uma com a outra |
| `charmm_polar_2013/2019/2023.xml` | CHARMM polarizável (osciladores de Drude) | Requer `Modeller.addExtraParticles` e integrador de Drude |
| `amoeba2009/2013/2018.xml`, `iamoeba.xml` | AMOEBA, polarizável com multipolos | Muito mais caro; sem restrições de ligação |
| `implicit/hct.xml`, `obc1.xml`, `obc2.xml`, `gbn.xml`, `gbn2.xml` | Solvente implícito Generalized Born (igb=1,2,5,7,8 do AMBER) | Adicionar ao FF Amber/CHARMM em vez do arquivo de água |
| `tip3p.xml`, `tip3pfb.xml`, `tip4pew.xml`, `tip4pfb.xml`, `tip5p.xml`, `spce.xml`, `opc.xml`, `opc3.xml`, `swm4ndp.xml` | Modelos de água (raiz do pacote) | As versões dentro de `amber14/`, `amber19/`, `charmm36*/` incluem também os íons da família |

**Além dos arquivos incluídos**, o pacote separado
[`openmmforcefields`](https://github.com/openmm/openmmforcefields) adiciona GAFF, OpenFF
(Sage/Parsley) e portes completos de Amber/CHARMM, com geração automática de parâmetros para
moléculas pequenas (ligantes). E `ForceField` aceita qualquer caminho para um XML seu.
""")

code(r"""
# =============================================================================
# Seção 2b — Lista dinâmica de TODOS os campos de força XML instalados com o OpenMM
# =============================================================================
# Os XML ficam em openmm/app/data. Percorremos essa pasta e agrupamos por subpasta.
data_dir = Path(app.__file__).parent / "data"
xml_files = sorted(p.relative_to(data_dir) for p in data_dir.rglob("*.xml"))

# Arquivos auxiliares que NÃO são campos de força (definições de nomes/resíduos/hidrogênios)
auxiliary = {"hydrogens.xml", "glycam-hydrogens.xml", "pdbNames.xml", "residues.xml"}

groups = {}
for rel in xml_files:
    if rel.name in auxiliary:
        continue
    groups.setdefault(str(rel.parent) if str(rel.parent) != "." else "(raiz)", []).append(str(rel))

print(f"OpenMM {openmm.__version__} — {sum(len(v) for v in groups.values())} arquivos de campo "
      f"de força em {data_dir}\n")
for folder, files in groups.items():
    print(f"[{folder}]")
    for f in files:
        print(f"   {f}")
    print()

print("Uso: app.ForceField('<biomoléculas>.xml', '<água+íons>.xml')  ->  ex.:",
      f"app.ForceField('{FF_PROTEIN}', '{FF_SOLVENT}')")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Baixar a estrutura e visualizar

Baixamos o arquivo PDB diretamente do RCSB [15]. Como 1L2Y é um *ensemble* de RMN com 38
modelos, extraímos apenas um (o modelo 1) e o gravamos em um PDB separado, para que **todas** as
etapas seguintes partam explicitamente da mesma coordenada.
""")

code(r"""
# =============================================================================
# Seção 3a — Download do PDB e extração de um único modelo
# =============================================================================
raw_pdb = OUT / f"{PDB_ID}_raw.pdb"
url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
if not raw_pdb.exists():
    print("Baixando", url)
    urllib.request.urlretrieve(url, raw_pdb)
print(f"Arquivo: {raw_pdb} ({raw_pdb.stat().st_size/1024:.1f} kB)")

# Cabeçalho: método experimental, título e número de modelos
with open(raw_pdb) as fh:
    lines = fh.readlines()
for tag in ("HEADER", "TITLE ", "EXPDTA"):
    for ln in lines:
        if ln.startswith(tag):
            print(ln.rstrip())
n_models = sum(1 for ln in lines if ln.startswith("MODEL"))
print(f"Número de modelos no ensemble: {n_models}")

# app.PDBFile lê todos os modelos como "frames" de posições sobre a mesma topologia.
pdb_all = app.PDBFile(str(raw_pdb))
print(f"Frames lidos: {pdb_all.getNumFrames()} | átomos por modelo: {pdb_all.topology.getNumAtoms()}")

# Gravamos SOMENTE o modelo escolhido.
model_pdb = OUT / f"{PDB_ID}_model{NMR_MODEL + 1}.pdb"
with open(model_pdb, "w") as fh:
    app.PDBFile.writeFile(pdb_all.topology, pdb_all.getPositions(frame=NMR_MODEL), fh)

seq = " ".join(res.name for res in pdb_all.topology.residues())
print(f"Sequência ({pdb_all.topology.getNumResidues()} resíduos): {seq}")
print("Modelo gravado em", model_pdb)
""")

code(r"""
# =============================================================================
# Seção 3b — Função de visualização com py3Dmol (reutilizada ao longo do notebook)
# =============================================================================
def show_structure(pdb_path, style="cartoon+sticks", width=640, height=420,
                   show_water=False, show_ions=True, show_box=False):
    '''Exibe um arquivo PDB em 3D dentro do notebook.

    style      : "cartoon+sticks" (proteína) ou "cartoon" (só o esqueleto)
    show_water : desenha as moléculas de água como linhas finas
    show_ions  : desenha íons como esferas
    show_box   : desenha a célula unitária (exige linha CRYST1 no PDB)
    '''
    with open(pdb_path) as fh:
        pdb_text = fh.read()
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")
    # Proteína: fita colorida por índice de resíduo (do azul, N-terminal, ao vermelho, C-terminal)
    view.setStyle({"resn": ["HOH", "WAT", "NA", "CL", "Na+", "Cl-"], "invert": True},
                  {"cartoon": {"color": "spectrum"}})
    if "sticks" in style:
        view.addStyle({"resn": ["HOH", "WAT", "NA", "CL"], "invert": True},
                      {"stick": {"radius": 0.15}})
    # Destaque do Trp6, o resíduo que dá nome à "gaiola de triptofano"
    view.addStyle({"resn": "TRP"}, {"stick": {"radius": 0.3, "colorscheme": "magentaCarbon"}})
    if show_water:
        view.addStyle({"resn": ["HOH", "WAT"]}, {"line": {"opacity": 0.5}})
    if show_ions:
        view.addStyle({"resn": ["NA", "CL"]}, {"sphere": {"radius": 0.8}})
    if show_box:
        view.addUnitCell()
    view.zoomTo()
    return view.show()

print("Modelo 1 de 1L2Y, direto do RCSB (com os hidrogênios do refinamento de RMN):")
show_structure(model_pdb)
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Preparar a proteína com PDBFixer

Arquivos do PDB raramente estão prontos para simular: podem faltar resíduos (alças não
resolvidas), átomos de cadeia lateral, hidrogênios (em cristalografia quase sempre faltam), e
podem conter resíduos não-padrão ou heteroátomos (ligantes, íons cristalográficos). O
[PDBFixer](https://github.com/openmm/pdbfixer) [16] resolve cada caso com uma chamada.

Para 1L2Y (RMN) esperamos **nenhum** resíduo ou átomo pesado faltante; os hidrogênios já
existem, mas mesmo assim chamamos `addMissingHydrogens(pH)`: ela mantém os H existentes, adiciona
os que faltarem e escolhe o **estado de protonação** de cada resíduo ionizável para o pH dado
(Asp/Glu desprotonados, Lys/Arg protonados, His conforme a rede de ligações de H). Imprimimos o
relatório de cada etapa para deixar registrado o que foi (e o que não foi) alterado.
""")

code(r"""
# =============================================================================
# Seção 4 — PDBFixer: completar e protonar a estrutura
# =============================================================================
from pdbfixer import PDBFixer

fixer = PDBFixer(filename=str(model_pdb))
n_atoms_in = fixer.topology.getNumAtoms()

# (1) Resíduos ausentes: compara SEQRES (sequência depositada) com os resíduos resolvidos.
fixer.findMissingResidues()
print("Resíduos faltantes :", dict(fixer.missingResidues) or "nenhum")

# (2) Resíduos não-padrão (ex.: MSE = selenometionina) → substituídos pelo equivalente padrão.
fixer.findNonstandardResidues()
print("Resíduos não-padrão:", fixer.nonstandardResidues or "nenhum")
fixer.replaceNonstandardResidues()

# (3) Heteroátomos (ligantes, íons, água cristalográfica). Aqui removemos tudo, inclusive água,
#     porque vamos solvatar do zero. Para manter um ligante, NÃO chame esta função e forneça
#     parâmetros para ele (ex. via openmmforcefields/GAFF).
fixer.removeHeterogens(keepWater=False)

# (4) Átomos pesados ausentes (cadeias laterais truncadas, OXT terminal).
fixer.findMissingAtoms()
print("Átomos faltantes   :", {str(k): v for k, v in fixer.missingAtoms.items()} or "nenhum")
print("Átomos terminais   :", {str(k): v for k, v in fixer.missingTerminals.items()} or "nenhum")
fixer.addMissingAtoms()

# (5) Hidrogênios coerentes com o pH escolhido.
fixer.addMissingHydrogens(PH)

fixed_pdb = OUT / f"{PDB_ID}_fixed.pdb"
with open(fixed_pdb, "w") as fh:
    app.PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

# Relatório final: contagem de átomos e estados de protonação escolhidos
n_atoms_out = fixer.topology.getNumAtoms()
print(f"\nÁtomos: {n_atoms_in} (entrada) → {n_atoms_out} (após PDBFixer)")
print("Resíduos após o preparo (nomes indicam variantes de protonação, ex. HID/HIE/HIP, ASH, GLH, LYN):")
print("  " + " ".join(f"{r.name}{r.id}" for r in fixer.topology.residues()))
print("Gravado em", fixed_pdb)
""")

code(r"""
# Proteína preparada (agora com todos os hidrogênios definidos pelo PDBFixer).
show_structure(fixed_pdb)
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Montar o sistema: campo de força, caixa de água e íons

1. `ForceField(FF_PROTEIN, FF_SOLVENT)` carrega os parâmetros.
2. `Modeller` é o editor de topologia do OpenMM. `addSolvent` calcula a esfera envolvente da
   proteína, cria uma caixa de aresta (diâmetro + *padding*), preenche com água (evitando
   sobreposição com a proteína; a densidade inicial fica um pouco abaixo de 1 g/mL, o barostato
   corrige isso na fase NPT),
   adiciona íons para **neutralizar** a carga líquida e depois íons extras até a **força iônica**
   pedida.
3. `createSystem` transforma topologia + FF em um objeto `System`: a lista de partículas,
   forças (ligações, ângulos, diedros, LJ + PME) e restrições (*constraints*).

Também gravamos o PDB do sistema completo, que servirá de **topologia para o MDAnalysis**.
""")

code(r"""
# =============================================================================
# Seção 5a — Carregar o campo de força e solvatar
# =============================================================================
forcefield = app.ForceField(FF_PROTEIN, FF_SOLVENT)

protein = app.PDBFile(str(fixed_pdb))
modeller = app.Modeller(protein.topology, protein.positions)
n_protein_atoms = modeller.topology.getNumAtoms()

t0 = time.time()
modeller.addSolvent(
    forcefield,
    model=WATER_MODEL,                       # geometria das moléculas de água inseridas
    padding=BOX_PADDING_NM * unit.nanometer, # aresta = diâmetro da proteína + padding
    boxShape=BOX_SHAPE,
    ionicStrength=IONIC_STRENGTH_M * unit.molar,
    positiveIon=POSITIVE_ION,
    negativeIon=NEGATIVE_ION,
    neutralize=True,                         # adiciona contra-íons para carga líquida zero
)
print(f"Solvatação em {time.time() - t0:.1f} s")

# ----- Resumo do que foi construído -------------------------------------------
from collections import Counter
counts = Counter(res.name for res in modeller.topology.residues())
n_water = counts.get("HOH", 0)
ions = {k: v for k, v in counts.items() if k in ("NA", "CL", "K", "LI", "CS", "RB", "BR", "F", "I")}
box = modeller.topology.getPeriodicBoxVectors()
box_nm = [v.value_in_unit(unit.nanometer) for v in box]
print(f"Átomos da proteína : {n_protein_atoms}")
print(f"Moléculas de água  : {n_water}")
print(f"Íons               : {ions}")
print(f"Átomos totais      : {modeller.topology.getNumAtoms()}")
print("Vetores da caixa (nm):")
for v in box_nm:
    print(f"   ({v.x:6.3f}, {v.y:6.3f}, {v.z:6.3f})")

# PDB do sistema completo: será a topologia lida pelo MDAnalysis na Seção 9.
solvated_pdb = OUT / f"{PDB_ID}_solvated.pdb"
with open(solvated_pdb, "w") as fh:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
print("Sistema gravado em", solvated_pdb)
""")

code(r"""
# Sistema solvatado: proteína em fita, água em linhas finas, íons como esferas, célula unitária.
show_structure(solvated_pdb, style="cartoon", show_water=True, show_ions=True, show_box=True,
               width=700, height=520)
""")

code(r"""
# =============================================================================
# Seção 5b — Criar o System (parâmetros de todas as interações)
# =============================================================================
system_kwargs = dict(
    nonbondedMethod=app.PME,                              # Ewald por malha p/ eletrostática
    nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer, # raio de corte real
    ewaldErrorTolerance=EWALD_ERROR_TOL,
    constraints=app.HBonds,     # ligações X–H rígidas (remove a vibração mais rápida)
    rigidWater=True,            # água rígida (padrão; explícito por clareza)
    removeCMMotion=True,        # remove deriva do centro de massa a cada passo
)
if USE_HMR:
    system_kwargs["hydrogenMass"] = HYDROGEN_MASS_AMU * unit.amu   # HMR (Hopkins et al. 2015)

system = forcefield.createSystem(modeller.topology, **system_kwargs)

# ----- Inspeção do System ------------------------------------------------------
print(f"Partículas : {system.getNumParticles()}")
print(f"Restrições : {system.getNumConstraints()} (ligações X–H + geometria da água)")
print("Forças     :")
for i, force in enumerate(system.getForces()):
    print(f"   [{i}] {force.__class__.__name__}")

# Carga total: soma das cargas parciais no NonbondedForce. Deve ser ~0 após a neutralização.
nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
total_charge = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                   for i in range(nb.getNumParticles()))
print(f"Carga total do sistema: {total_charge:+.4f} e  (esperado ≈ 0)")
if abs(total_charge) > 1e-3:
    raise RuntimeError("Sistema não está neutro; verifique os íons/FF.")

# Conferência do HMR: massa de um H da proteína e massa total (que deve ser preservada)
if USE_HMR:
    h_index = next(a.index for a in modeller.topology.atoms()
                   if a.element is not None and a.element.symbol == "H")
    print(f"Massa de um H após HMR: {system.getParticleMass(h_index).value_in_unit(unit.amu):.3f} u")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Minimização de energia

As coordenadas iniciais têm sobreposições água–proteína e pequenas tensões geométricas que
gerariam forças enormes no primeiro passo de MD. O minimizador L-BFGS do OpenMM leva o sistema
ao mínimo local mais próximo. Reportamos a energia potencial antes/depois e o quanto os átomos
pesados da proteína se moveram (deve ser pequeno: minimização não muda a conformação).
""")

code(r"""
# =============================================================================
# Seção 6 — Minimização
# =============================================================================
integrator_min = openmm.VerletIntegrator(1.0 * unit.femtosecond)  # não é usado para dinâmica;
                                                                  # a minimização só precisa de um Context
sim_min = app.Simulation(modeller.topology, system, integrator_min, PLATFORM, PLATFORM_PROPERTIES)
sim_min.context.setPositions(modeller.positions)

def potential_energy(simulation):
    # Energia potencial em kJ/mol do estado atual do Context.
    return simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)

e_before = potential_energy(sim_min)
t0 = time.time()
sim_min.minimizeEnergy(tolerance=MINIMIZE_TOLERANCE * unit.kilojoule_per_mole / unit.nanometer)
e_after = potential_energy(sim_min)
print(f"Energia potencial: {e_before:14.1f} kJ/mol (antes)")
print(f"                   {e_after:14.1f} kJ/mol (depois)   Δ = {e_after - e_before:.1f} kJ/mol")
print(f"Tempo: {time.time() - t0:.1f} s na plataforma {PLATFORM.getName()}")

# Posições minimizadas (com moléculas inteiras dentro da caixa) → PDB de referência
state_min = sim_min.context.getState(getPositions=True, enforcePeriodicBox=True)
minimized_positions = state_min.getPositions()
minimized_pdb = OUT / "minimized.pdb"
with open(minimized_pdb, "w") as fh:
    app.PDBFile.writeFile(modeller.topology, minimized_positions, fh)

# Deslocamento RMS dos átomos pesados da proteína durante a minimização
# A proteína ocupa os primeiros n_protein_atoms índices (o Modeller acrescentou água/íons depois).
heavy_idx = [a.index for a in modeller.topology.atoms()
             if a.index < n_protein_atoms and a.element is not None and a.element.symbol != "H"]
xyz0 = np.array(modeller.positions.value_in_unit(unit.nanometer))[heavy_idx]
xyz1 = np.array(sim_min.context.getState(getPositions=True).getPositions()
                .value_in_unit(unit.nanometer))[heavy_idx]
rmsd_min = np.sqrt(((xyz1 - xyz0) ** 2).sum(axis=1).mean())
print(f"Deslocamento RMS dos {len(heavy_idx)} átomos pesados da proteína: {rmsd_min*10:.3f} Å "
      "(sem superposição: inclui qualquer translação/rotação de corpo rígido)")
print("Estrutura minimizada gravada em", minimized_pdb)
del sim_min  # libera o Context da GPU
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Equilibração NVT (1 ns)

Objetivo: levar o sistema a 300 K e deixar o solvente relaxar em torno da proteína, **sem** que
a proteína se afaste da estrutura experimental antes de estar cercada por água equilibrada.
Para isso:

- atribuímos velocidades iniciais de Maxwell–Boltzmann a 300 K (`setVelocitiesToTemperature`);
- o termostato de Langevin mantém a temperatura (o "N-V-T": número de partículas, volume e
  temperatura constantes; o volume é o da caixa criada na Seção 5);
- uma **restrição harmônica** prende cada átomo pesado da proteína à sua posição minimizada. Ela
  é implementada como `CustomExternalForce` com energia `k·d²`, onde `d` é a distância
  periódica ao ponto de referência.

Um `StateDataReporter` grava temperatura, energias e velocidade (ns/dia) em CSV; outro imprime
o progresso na tela. A trajetória vai para um arquivo XTC (formato compacto do GROMACS, lido
pelo MDAnalysis). No fim, gravamos o estado completo (posições, velocidades, caixa) em XML.
""")

code(r"""
# =============================================================================
# Seção 7a — Utilitários: conversão ps → passos e criação dos reporters
# =============================================================================
def ps_to_steps(interval_ps):
    # Converte um intervalo em ps para número inteiro de passos (mínimo 1).
    return max(1, int(round(interval_ps * 1000.0 / TIMESTEP_FS)))

def add_reporters(simulation, prefix, total_steps):
    '''Anexa reporters padrão a uma Simulation. `prefix` nomeia os arquivos (nvt, npt).'''
    simulation.reporters.clear()
    # Trajetória (XTC): posições, com moléculas re-embrulhadas inteiras dentro da caixa.
    simulation.reporters.append(app.XTCReporter(str(OUT / f"{prefix}.xtc"),
                                                ps_to_steps(TRAJ_INTERVAL_PS),
                                                enforcePeriodicBox=True))
    # Observáveis em CSV (uma linha por intervalo): base das figuras da Seção 9.
    simulation.reporters.append(app.StateDataReporter(
        str(OUT / f"{prefix}.csv"), ps_to_steps(CSV_INTERVAL_PS),
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True, separator=","))
    # Progresso na tela.
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, ps_to_steps(STDOUT_INTERVAL_PS), step=True, time=True,
        potentialEnergy=True, temperature=True, density=True, speed=True,
        progress=True, remainingTime=True, totalSteps=total_steps, separator="\t"))
    # Checkpoint binário: permite retomar com simulation.loadCheckpoint(...)
    simulation.reporters.append(app.CheckpointReporter(str(OUT / f"{prefix}.chk"),
                                                       ps_to_steps(CHECKPOINT_INTERVAL_PS)))

def make_integrator():
    # Integrador de Langevin "middle" (Zhang et al. 2019): termostato embutido.
    integ = openmm.LangevinMiddleIntegrator(TEMPERATURE_K * unit.kelvin,
                                            FRICTION_PER_PS / unit.picosecond,
                                            TIMESTEP_FS * unit.femtosecond)
    integ.setRandomNumberSeed(RANDOM_SEED)
    return integ

print(f"1 ps = {ps_to_steps(1.0)} passos de {TIMESTEP_FS} fs")
""")

code(r"""
# =============================================================================
# Seção 7b — Restrições harmônicas nos átomos pesados da proteína
# =============================================================================
# E = k * periodicdistance(x,y,z, x0,y0,z0)^2   (k é parâmetro global; x0,y0,z0 por átomo)
restraint = openmm.CustomExternalForce("k_restr*periodicdistance(x, y, z, x0, y0, z0)^2")
restraint.addGlobalParameter("k_restr", RESTRAINT_K * unit.kilojoule_per_mole / unit.nanometer**2)
for name in ("x0", "y0", "z0"):
    restraint.addPerParticleParameter(name)

ref_xyz = minimized_positions  # posições de referência = estrutura minimizada
for idx in heavy_idx:
    restraint.addParticle(idx, ref_xyz[idx].value_in_unit(unit.nanometer))

restraint_index = system.addForce(restraint)   # guardamos o índice para remover na Seção 8
print(f"Restrição harmônica (k = {RESTRAINT_K} kJ/mol/nm²) aplicada a {restraint.getNumParticles()} "
      f"átomos pesados da proteína; índice da força no System: {restraint_index}")
""")

code(r"""
# =============================================================================
# Seção 7c — Rodar a fase NVT
# =============================================================================
nvt_steps = ps_to_steps(NVT_NS * 1000.0)

sim_nvt = app.Simulation(modeller.topology, system, make_integrator(), PLATFORM, PLATFORM_PROPERTIES)
sim_nvt.context.setPositions(minimized_positions)
sim_nvt.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, RANDOM_SEED)

add_reporters(sim_nvt, "nvt", nvt_steps)
print(f"NVT: {NVT_NS} ns = {nvt_steps} passos a {TIMESTEP_FS} fs, {TEMPERATURE_K} K, "
      f"caixa fixa, proteína restrita\n")

t0 = time.time()
sim_nvt.step(nvt_steps)
wall = time.time() - t0

# Estado final (posições, velocidades, vetores da caixa) → XML legível
nvt_state = sim_nvt.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
sim_nvt.saveState(str(OUT / "nvt_final.xml"))
with open(OUT / "nvt_final.pdb", "w") as fh:
    app.PDBFile.writeFile(modeller.topology, nvt_state.getPositions(), fh)

print(f"\nNVT concluída em {wall/60:.1f} min  →  {NVT_NS / (wall / 86400):.1f} ns/dia "
      f"na plataforma {PLATFORM.getName()}")
del sim_nvt
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Produção NPT (5 ns)

Agora o sistema evolui livremente a **temperatura e pressão** constantes, como no experimento:

- removemos a força de restrição do `System`;
- adicionamos o barostato de Monte Carlo (`MonteCarloBarostat`): a cada 25 passos ele propõe uma
  mudança isotrópica de volume e aceita/rejeita pelo critério de Metropolis [9, 10]. O barostato
  **não** controla a temperatura; por isso o integrador de Langevin continua ativo e ambos usam
  a mesma temperatura;
- como o `System` mudou, criamos uma **nova** `Simulation` e carregamos nela posições,
  velocidades e caixa do final da fase NVT (continuidade da dinâmica).

A densidade agora flutua e deve convergir para ~1 g cm⁻³ nas primeiras centenas de ps. Para
análises quantitativas, esse trecho inicial deve ser tratado como equilibração de pressão e
descartado (ver Seção 9).
""")

code(r"""
# =============================================================================
# Seção 8 — Rodar a fase NPT
# =============================================================================
# (1) remover a restrição harmônica
removed = system.getForce(restraint_index).__class__.__name__
system.removeForce(restraint_index)
print(f"Força removida do System: {removed}")

# (2) adicionar o barostato de Monte Carlo
barostat = openmm.MonteCarloBarostat(PRESSURE_BAR * unit.bar, TEMPERATURE_K * unit.kelvin, 25)
barostat.setRandomNumberSeed(RANDOM_SEED)
barostat_index = system.addForce(barostat)
print("Forças agora:", [f.__class__.__name__ for f in system.getForces()])

# (3) nova Simulation, continuando exatamente de onde a NVT parou
npt_steps = ps_to_steps(NPT_NS * 1000.0)
sim_npt = app.Simulation(modeller.topology, system, make_integrator(), PLATFORM, PLATFORM_PROPERTIES)
sim_npt.context.setState(nvt_state)     # posições + velocidades + caixa da fase NVT
# O State também carrega o tempo e o contador de passos da NVT. Zeramos ambos para que o
# relógio da fase NPT (CSV, XTC, barra de progresso) comece em 0.
sim_npt.context.setTime(0.0)
sim_npt.context.setStepCount(0)

add_reporters(sim_npt, "npt", npt_steps)
print(f"\nNPT: {NPT_NS} ns = {npt_steps} passos a {TIMESTEP_FS} fs, {TEMPERATURE_K} K, "
      f"{PRESSURE_BAR} bar, sem restrições\n")

t0 = time.time()
sim_npt.step(npt_steps)
wall = time.time() - t0

sim_npt.saveState(str(OUT / "npt_final.xml"))
npt_state = sim_npt.context.getState(getPositions=True, enforcePeriodicBox=True)
final_pdb = OUT / "npt_final.pdb"
with open(final_pdb, "w") as fh:
    app.PDBFile.writeFile(modeller.topology, npt_state.getPositions(), fh)

print(f"\nNPT concluída em {wall/60:.1f} min  →  {NPT_NS / (wall / 86400):.1f} ns/dia "
      f"na plataforma {PLATFORM.getName()}")
print("Arquivos gerados:")
for p in sorted(OUT.iterdir()):
    print(f"   {p.name:28s} {p.stat().st_size/1e6:8.2f} MB")
del sim_npt
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Análise

### 9a. Observáveis termodinâmicos (dos CSV do `StateDataReporter`)

O que verificar:
- **Temperatura** flutuando em torno de 300 K nas duas fases (a amplitude das flutuações
  instantâneas é ~ T·√(2/3N_gl), pequena para milhares de átomos). Nos primeiros picossegundos
  da NVT a temperatura cai abaixo do alvo: as velocidades foram sorteadas a 300 K sobre uma
  estrutura minimizada, e parte da energia cinética vira energia potencial (equipartição); o
  termostato repõe a diferença rapidamente.
- **Densidade** constante na NVT (volume fixo; fica abaixo de 1 g cm⁻³ porque `addSolvent`
  empacota a água com folga) e, na NPT, convergindo para ~1 g cm⁻³. O tempo que ela leva para
  estabilizar define o trecho a descartar como equilibração.
- **Energia potencial** sem deriva sistemática após a equilibração.

Cada gráfico tem um único eixo y (uma grandeza por painel).
""")

code(r"""
# =============================================================================
# Seção 9a — Séries temporais dos observáveis
# =============================================================================
COLORS = {"nvt": "#2a78d6", "npt": "#eb6834"}   # azul = NVT, laranja = NPT (paleta fixa)

nvt_df = pd.read_csv(OUT / "nvt.csv")
npt_df = pd.read_csv(OUT / "npt.csv")
for df in (nvt_df, npt_df):
    # a primeira linha de 'Speed' pode vir como texto ("--"); força tudo a numérico
    df["Speed (ns/day)"] = pd.to_numeric(df["Speed (ns/day)"], errors="coerce")
# Colunas: '#"Step"', 'Time (ps)', 'Potential Energy (kJ/mole)', ..., 'Temperature (K)',
#          'Box Volume (nm^3)', 'Density (g/mL)', 'Speed (ns/day)'
nvt_df["t_ns"] = nvt_df["Time (ps)"] / 1000.0
# A NPT continua no tempo depois da NVT: deslocamos o eixo para começar em NVT_NS.
npt_df["t_ns"] = nvt_df["t_ns"].iloc[-1] + npt_df["Time (ps)"] / 1000.0

panels = [("Temperature (K)", "Temperatura (K)"),
          ("Density (g/mL)", "Densidade (g/mL)"),
          ("Potential Energy (kJ/mole)", "Energia potencial (kJ/mol)"),
          ("Box Volume (nm^3)", "Volume da caixa (nm³)")]

fig, axes = plt.subplots(len(panels), 1, figsize=(9, 10), sharex=True)
for ax, (col, label) in zip(axes, panels):
    ax.plot(nvt_df["t_ns"], nvt_df[col], lw=1, color=COLORS["nvt"], label="NVT (restrita)")
    ax.plot(npt_df["t_ns"], npt_df[col], lw=1, color=COLORS["npt"], label="NPT (produção)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
axes[0].axhline(TEMPERATURE_K, color="gray", lw=0.8, ls="--")
axes[0].legend(loc="upper right", frameon=False)
axes[-1].set_xlabel("tempo de simulação (ns)")
fig.suptitle(f"{PDB_ID}: observáveis do StateDataReporter", y=0.995)
fig.tight_layout()
plt.show()

# Estatísticas simples (média ± desvio-padrão) por fase — n é o número de amostras no CSV.
def summarize(df, name):
    print(f"{name}: n = {len(df)} amostras (uma a cada {CSV_INTERVAL_PS} ps)")
    for col, label in panels:
        print(f"   {label:28s} {df[col].mean():12.3f} ± {df[col].std():.3f}")
    print(f"   {'Velocidade média (ns/dia)':28s} {df['Speed (ns/day)'].iloc[1:].mean():12.1f}")
summarize(nvt_df, "NVT")
summarize(npt_df, "NPT")
""")

md(r"""
### 9b. Análise estrutural com MDAnalysis

Carregamos a trajetória NPT com o PDB do sistema solvatado como topologia. O `XTCReporter` já
gravou as moléculas inteiras dentro da caixa, então a proteína nunca aparece "quebrada".

- **RMSD** (*root-mean-square deviation*, desvio quadrático médio) do esqueleto em relação à
  estrutura minimizada, após superposição ótima: mede quanto a conformação se afastou do ponto
  de partida.
- **RMSF** (*root-mean-square fluctuation*) por resíduo (átomos Cα): mede a flexibilidade local.
  Segue o procedimento recomendado pelo MDAnalysis: alinha ao primeiro frame, calcula a
  estrutura média, realinha à média e só então calcula a flutuação.
- **Raio de giro** (R_g): compacidade da proteína ao longo do tempo.
""")

code(r"""
# =============================================================================
# Seção 9b — RMSD, RMSF e raio de giro com MDAnalysis
# =============================================================================
from MDAnalysis.analysis import rms, align

u = mda.Universe(str(solvated_pdb), str(OUT / "npt.xtc"))
ref = mda.Universe(str(minimized_pdb))          # referência: estrutura minimizada
protein_sel = "protein"
prot = u.select_atoms(protein_sel)
print(f"Universo: {u.atoms.n_atoms} átomos, {len(u.trajectory)} frames, "
      f"dt = {u.trajectory.dt:.2f} ps, proteína = {prot.n_atoms} átomos / {prot.n_residues} resíduos")

# ----- RMSD (superposição pelo backbone; coluna extra: átomos pesados da proteína) --------
R = rms.RMSD(u, ref, select="protein and backbone",
             groupselections=["protein and not name H*"]).run()
rmsd = R.results.rmsd            # colunas: frame, tempo (ps), RMSD backbone, RMSD pesados (Å)
t_ns = rmsd[:, 1] / 1000.0

# ----- Raio de giro ---------------------------------------------------------------------
rg = np.array([prot.radius_of_gyration() for ts in u.trajectory])   # Å

# ----- RMSF por resíduo (Cα) --------------------------------------------------------------
# (1) alinhar todos os frames ao primeiro (em memória, cópia da trajetória)
align.AlignTraj(u, u, select="protein and name CA", in_memory=True).run()
# (2) estrutura média e (3) realinhamento à média
avg = align.AverageStructure(u, u, select="protein and name CA", ref_frame=0).run()
ref_avg = avg.results.universe
align.AlignTraj(u, ref_avg, select="protein and name CA", in_memory=True).run()
# (4) RMSF dos Cα
ca = u.select_atoms("protein and name CA")
rmsf = rms.RMSF(ca).run().results.rmsf

# ----- Figuras --------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
axes[0].plot(t_ns, rmsd[:, 2], lw=1, color=COLORS["npt"], label="backbone")
axes[0].plot(t_ns, rmsd[:, 3], lw=1, color="#1baf7a", label="átomos pesados")
axes[0].set_ylabel("RMSD vs. minimizada (Å)"); axes[0].set_xlabel("tempo NPT (ns)")
axes[0].legend(frameon=False)
axes[1].plot(t_ns, rg, lw=1, color=COLORS["npt"])
axes[1].set_ylabel("Raio de giro (Å)"); axes[1].set_xlabel("tempo NPT (ns)")
axes[2].bar(ca.resids, rmsf, color=COLORS["npt"], width=0.8)
axes[2].set_xticks(ca.resids)
axes[2].set_xticklabels([f"{r.resname[:3].title()}{r.resid}" for r in ca.residues], rotation=60)
axes[2].set_ylabel("RMSF Cα (Å)")
for ax in axes:
    ax.grid(alpha=0.25)
fig.suptitle(f"{PDB_ID}: análise estrutural da fase NPT ({len(u.trajectory)} frames)", y=0.995)
fig.tight_layout()
plt.show()

# ----- Números (com n) -------------------------------------------------------------------
print(f"RMSD backbone : média {rmsd[:,2].mean():.2f} Å, máx {rmsd[:,2].max():.2f} Å "
      f"(n = {len(rmsd)} frames)")
print(f"Raio de giro  : média {rg.mean():.2f} ± {rg.std():.2f} Å")
print("RMSF por resíduo (Å): " + ", ".join(f"{r.resname}{r.resid}={v:.2f}" for r, v in zip(ca.residues, rmsf)))
""")

md(r"""
### 9c. Visualizar a trajetória e comparar início × fim

O `py3Dmol` anima um PDB com vários modelos. Usamos os frames já alinhados pelo MDAnalysis,
apenas a proteína, com no máximo ~50 frames para manter o arquivo leve. Em seguida, sobrepomos a
estrutura minimizada (cinza) e o último frame (colorido).
""")

code(r"""
# =============================================================================
# Seção 9c — Animação da proteína (frames alinhados) e sobreposição início/fim
# =============================================================================
import warnings
# Ao gravar PDB, o MDAnalysis avisa que não há cargas formais nem CRYST1 no recorte da proteína.
# São avisos cosméticos (não afetam coordenadas); silenciamos apenas esses dois.
warnings.filterwarnings("ignore", message=".*formalcharges.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Unit cell dimensions not found.*", category=UserWarning)

stride = max(1, len(u.trajectory) // 50)          # ≤ ~50 frames na animação
frames_pdb = OUT / "npt_protein_aligned.pdb"
with mda.Writer(str(frames_pdb), multiframe=True, n_atoms=prot.n_atoms) as w:
    for ts in u.trajectory[::stride]:
        w.write(prot)
print(f"{len(u.trajectory[::stride])} frames (1 a cada {stride}) gravados em {frames_pdb}")

view = py3Dmol.view(width=640, height=420)
view.addModelsAsFrames(open(frames_pdb).read(), "pdb")
view.setStyle({}, {"cartoon": {"color": "spectrum"}})
view.addStyle({"resn": "TRP"}, {"stick": {"radius": 0.3, "colorscheme": "magentaCarbon"}})
view.zoomTo()
view.animate({"loop": "forward", "interval": 120})   # ms entre frames
view.show()
""")

code(r"""
# Sobreposição: estrutura minimizada (cinza) × último frame da NPT (colorido), alinhados pelo Cα
last = mda.Universe(str(frames_pdb))
last.trajectory[-1]                      # posiciona no último frame gravado
last_prot = last.select_atoms("protein")
ref_prot = ref.select_atoms("protein")
# superposição ótima (Kabsch) do último frame sobre a referência, usando os Cα
align.alignto(last_prot, ref_prot, select="protein and name CA")

ref_prot.write(str(OUT / "_ref_prot.pdb"))
last_prot.write(str(OUT / "_last_prot.pdb"))

view = py3Dmol.view(width=640, height=420)
view.addModel(open(OUT / "_ref_prot.pdb").read(), "pdb")
view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray"}})
view.addModel(open(OUT / "_last_prot.pdb").read(), "pdb")
view.setStyle({"model": 1}, {"cartoon": {"color": "spectrum"}})
view.addStyle({"model": 1, "resn": "TRP"}, {"stick": {"radius": 0.3, "colorscheme": "magentaCarbon"}})
view.zoomTo()
view.show()
print("Cinza: minimizada. Colorido: último frame da NPT. Trp6 em magenta.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. O que estes resultados permitem (e não permitem) concluir

- O notebook produz **uma** trajetória de produção de 5 ns (500 frames a cada 10 ps com os
  parâmetros padrão). Isso basta para verificar que o protocolo está correto (temperatura e
  densidade estáveis, RMSD sem explosão) e para ilustrar as análises.
- **Não** basta para afirmar que a Trp-cage é "estável" no campo de força escolhido: os tempos
  de enovelamento/desenovelamento da Trp-cage são de microssegundos [1], ou seja, ordens de
  grandeza maiores que a janela simulada. Um RMSD baixo em 5 ns é compatível tanto com uma
  estrutura estável quanto com uma que ainda não teve tempo de sair do mínimo inicial.
- Conclusões quantitativas exigem **replicatas independentes** (sementes diferentes), tempos
  muito maiores, e estimativas de erro por blocos ou por replicata. Frames consecutivos são
  correlacionados; `n = 500` frames não são 500 amostras independentes.
- Para trocar o sistema, basta alterar `PDB_ID`; para proteínas cristalográficas, o PDBFixer
  passa a ter trabalho real (hidrogênios, alças, heteroátomos) e vale inspecionar cada
  relatório da Seção 4.

### Salvar resultados (opcional, Colab)
Descomente para copiar a pasta de saída para o seu Google Drive, ou para baixar um zip.
""")

code(r"""
# # --- Opção A: Google Drive ---
# from google.colab import drive
# drive.mount("/content/drive")
# !cp -r {OUTPUT_DIR} /content/drive/MyDrive/

# # --- Opção B: baixar um zip ---
# import shutil
# from google.colab import files
# shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
# files.download(f"{OUTPUT_DIR}.zip")
""")

md(r"""
## Referências

1. Neidigh, J. W.; Fesinmeyer, R. M.; Andersen, N. H. Designing a 20-residue protein.
   *Nat. Struct. Biol.* **2002**, *9*, 425–430. [10.1038/nsb798](https://doi.org/10.1038/nsb798)
2. Maier, J. A. *et al.* ff14SB: Improving the accuracy of protein side chain and backbone
   parameters from ff99SB. *J. Chem. Theory Comput.* **2015**, *11*, 3696–3713.
   [10.1021/acs.jctc.5b00255](https://doi.org/10.1021/acs.jctc.5b00255)
3. Jorgensen, W. L. *et al.* Comparison of simple potential functions for simulating liquid
   water. *J. Chem. Phys.* **1983**, *79*, 926–935. [10.1063/1.445869](https://doi.org/10.1063/1.445869)
4. Tian, C. *et al.* ff19SB: Amino-acid-specific protein backbone parameters trained against
   quantum mechanics energy surfaces in solution. *J. Chem. Theory Comput.* **2020**, *16*,
   528–552. [10.1021/acs.jctc.9b00591](https://doi.org/10.1021/acs.jctc.9b00591)
5. Izadi, S.; Anandakrishnan, R.; Onufriev, A. V. Building water models: a different approach.
   *J. Phys. Chem. Lett.* **2014**, *5*, 3863–3871. [10.1021/jz501780a](https://doi.org/10.1021/jz501780a)
6. Essmann, U. *et al.* A smooth particle mesh Ewald method. *J. Chem. Phys.* **1995**, *103*,
   8577–8593. [10.1063/1.470117](https://doi.org/10.1063/1.470117)
7. Zhang, Z.; Liu, X.; Yan, K.; Tuckerman, M. E.; Liu, J. Unified efficient thermostat scheme for
   the canonical ensemble with holonomic or isokinetic constraints via molecular dynamics.
   *J. Phys. Chem. A* **2019**, *123*, 6056–6079. [10.1021/acs.jpca.9b02771](https://doi.org/10.1021/acs.jpca.9b02771)
8. Hopkins, C. W.; Le Grand, S.; Walker, R. C.; Roitberg, A. E. Long-time-step molecular dynamics
   through hydrogen mass repartitioning. *J. Chem. Theory Comput.* **2015**, *11*, 1864–1874.
   [10.1021/ct5010406](https://doi.org/10.1021/ct5010406)
9. Chow, K.-H.; Ferguson, D. M. Isothermal-isobaric molecular dynamics simulations with Monte
   Carlo volume sampling. *Comput. Phys. Commun.* **1995**, *91*, 283–289.
   [10.1016/0010-4655(95)00059-O](https://doi.org/10.1016/0010-4655(95)00059-O)
10. Åqvist, J.; Wennerström, P.; Nervall, M.; Bjelic, S.; Brandsdal, B. O. Molecular dynamics
    simulations of water and biomolecules with a Monte Carlo constant pressure algorithm.
    *Chem. Phys. Lett.* **2004**, *384*, 288–294.
    [10.1016/j.cplett.2003.12.039](https://doi.org/10.1016/j.cplett.2003.12.039)
11. Michaud-Agrawal, N.; Denning, E. J.; Woolf, T. B.; Beckstein, O. MDAnalysis: a toolkit for
    the analysis of molecular dynamics simulations. *J. Comput. Chem.* **2011**, *32*, 2319–2327.
    [10.1002/jcc.21787](https://doi.org/10.1002/jcc.21787)
12. Gowers, R. J. *et al.* MDAnalysis: a Python package for the rapid analysis of molecular
    dynamics simulations. *Proc. 15th Python in Science Conf.* **2016**, 98–105.
    [10.25080/Majora-629e541a-00e](https://doi.org/10.25080/Majora-629e541a-00e)
13. Rego, N.; Koes, D. 3Dmol.js: molecular visualization with WebGL. *Bioinformatics* **2015**,
    *31*, 1322–1324. [10.1093/bioinformatics/btu829](https://doi.org/10.1093/bioinformatics/btu829)
14. Huang, J. *et al.* CHARMM36m: an improved force field for folded and intrinsically disordered
    proteins. *Nat. Methods* **2017**, *14*, 71–73. [10.1038/nmeth.4067](https://doi.org/10.1038/nmeth.4067)
15. Berman, H. M. *et al.* The Protein Data Bank. *Nucleic Acids Res.* **2000**, *28*, 235–242.
    [10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235)
16. Eastman, P. *et al.* OpenMM 8: molecular dynamics simulation with machine learning
    potentials. *J. Phys. Chem. B* **2024**, *128*, 109–116.
    [10.1021/acs.jpcb.3c06662](https://doi.org/10.1021/acs.jpcb.3c06662). PDBFixer:
    [github.com/openmm/pdbfixer](https://github.com/openmm/pdbfixer). Manual do OpenMM:
    [docs.openmm.org](https://docs.openmm.org/latest/userguide/).
""")

nb.cells = cells
out = sys.argv[1] if len(sys.argv) > 1 else "openmm_1L2Y_colab.ipynb"
nbf.write(nb, out)
print(f"{out}: {len(cells)} células")
