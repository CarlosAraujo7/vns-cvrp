# Does adaptive operator selection help? Replication package

Code, instances, raw results and analysis scripts for the study *"Does adaptive operator selection help? A controlled,
pre-specified study of bandit-guided variable neighborhood search for the capacitated vehicle routing problem"*
(manuscript in `paper/ejor/`; fourteen studies on the CVRP, the TSP and the CVRP with time windows). The research log with the dated pre-specification of every study is
`../diario_pesquisa.md` (Portuguese; entries 64, 71, 77, 83, 89, 91 and 97 are the pre-registrations of Studies 1-7; entries 101, 103 (amended by 107), 104, 106, 109, 113 and 116 those of Studies 8, 9, 10, 11, 12, 13 and 14).

## Quick path: regenerate every table, number and figure of the manuscript from the shipped raw results

The raw per-run files of all studies are in `results/aos_study/`. From `paper/ejor/` (Windows or Linux Python with `numpy`, `pandas`, `scipy`, `matplotlib`; no solver needed):

```
python ../../estudo_aos/verificacao_independente.py   # recomputes means and contrasts with the standard library only (expects OK)
python gerar_tabelas.py        # tables (tab_*.tex), numeros.tex/json, suptab.tex
python gerar_figuras.py        # figures
python gerar_suplemento.py     # supplementary per-instance tables
python checar_macros.py        # every \num{} key exists; abstract length; no control bytes
python checar_afirmacoes.py    # universal claims (U1-U6) checked against the confirmatory tables
bash compilar.sh               # manuscript and supplement (Tectonic)
```

The analysis scripts (`estudo_aos/analisar_etapa*.py`) regenerate the `analise_*.csv` files from the raw runs; `python tests/validate_tsp_neighborhoods.py`, `python tests/validate_tw_neighborhoods.py` and `python tests/test_tsp_regret_equiv.py` re-run the operator validations (the first two include mutation tests and take a few minutes).

## Layout

| Path | Contents |
|---|---|
| `src/` | Core code: instance reader, construction, neighborhoods (`neighborhoods.py`: 5 original operators; `neighborhoods_ext.py`: 5 added operators), the reproduced HG-UCB bandit (`bandit.py`), search loops (`vns.py`, `aos_vnd.py`), classical AOS policies (`aos_policies.py`) and the additional policies of Study 6 (`aos_policies_ext.py`: EXP3, Thompson sampling, sliding-window UCB, adaptive pursuit, efficiency UCB), distance-evaluation counter (`instancia_contadora.py`), stored initial solutions loader (`solucao_inicial.py`) |
| `src/hgs_aos/` | Modified copy of HGS-CVRP (MIT license, notice in `src/hgs_aos/LICENSE`) with adaptive move-order policies, used in Study 7 (`direcao1_local/build_hgs_aos.sh` builds it) |
| `estudo_aos/` | Runners and analyses of Studies 1-7 (`run_aos_*.py`, `analisar_*.py`), robustness analysis, independent verification, synthetic-instance generator, initial-solution generator, structure diagnostic (`diag_estrutura.py`), Study 4 and 6 pipelines |
| `instances/` | CVRPLIB subsets `A`, `B`, `CMT`, `X` (74 instances, with reference solutions) and the synthetic sets `SYN_TUNE` (24) and `SYN_VAL` (30) with HGS-CVRP reference solutions and `metadata.csv` |
| `results/` | Raw per-run results (`aos_study/*.csv`, `hgs_74_*.csv`, `kool_zeroshot_74.csv`, ...) |
| `tests/` | Validation scripts; `validate_neighborhoods_ext.py` checks the added operators against brute-force enumeration |
| `direcao1_local/` | Scripts for the reference solvers (HGS-CVRP, OR-Tools, LKH-3) and the neural baseline |
| `paper/` | Manuscript, bibliography (only references verified against primary sources), provenance of every number (`RASTREIO_NUMEROS.md`) |

## Environment

Python 3.14.4 (NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, river 0.26.1 for ADWIN; see `requirements.txt`), Ubuntu 26.04 LTS on WSL2,
g++ 15.2.0. HGS-CVRP (MIT license, <https://github.com/vidalt/HGS-CVRP>, commit `1a92795`) is built with
`g++ -std=c++17 -O3 -DNDEBUG Program/*.cpp -o hgs` inside `external/HGS-CVRP/` (not redistributed here).
Budgets are operator applications or distance evaluations, so results do not depend on the hardware; they do depend
on the Python/NumPy versions only through floating-point-free integer arithmetic of the counters and the seeds
(`random.Random(seed)`).

## Reproducing the results

All commands run from `estudo_aos/` inside the Python environment (the experiments need `river`).

1. **Study 1** (original loop, 74 instances, 10 seeds, B = 800 and 4000, 8 arms): `python run_aos_study.py --full --workers 11` (about 8 h on 12 threads; resumable). Analysis: `python analisar_aos.py`.
2. **Study 2** (structure decomposition and AOS inside VND): `python run_aos_etapa2.py --full` (about 8 h); `python analisar_etapa2.py`.
3. **Study 3** (ten operators, distance-evaluation budgets): `python run_aos_etapa3.py --full` (about 11 h); `python analisar_etapa3.py`.
4. **Study 4** (tuning, static mixture, synthetic validation): `python gerar_instancias_sinteticas.py` (needs the HGS binary), then `bash etapa4_pipeline.sh` (about 6 h) and `python analisar_etapa4.py`.
5. **Study 5** (which operators explain the pool effect; uniform selection on operator subsets): `python run_aos_etapa5.py --full` (about 2 h); `python analisar_etapa5.py`.
6. **Study 6** (EXP3, Thompson sampling, sliding-window UCB, adaptive pursuit, efficiency UCB): `bash etapa6_pipeline.sh` (about 2 h; waits for Study 5 to finish) and `python analisar_etapa6.py`; unit tests of the new policies: `python ../tests/test_aos_policies_ext.py`.
7. **Initial solutions**: all arms start from the solution stored in `instances/_init/` (generated once by `python gerar_solucoes_iniciais.py`; the Clarke--Wright repair uses a non-deterministic CP-SAT fallback on 11 instances, which is why the solutions are persisted). **Structure diagnostic**: `python diag_estrutura.py`; **complementarity diagnostic** (exploratory): `python diag_complementaridade.py`.
8. **Robustness and verification**: `python analise_robustez.py` (equivalence tests, two-level bootstrap, power); `python verificacao_independente.py` (recomputes the manuscript means and contrasts from raw costs with the standard library only); `python ../tests/validate_neighborhoods_ext.py`.
9. **Study 7** (adaptive move order inside HGS-CVRP): `bash direcao1_local/build_hgs_aos.sh` (builds the modified copy in `src/hgs_aos/` to `external/HGS-CVRP-aos/hgs_aos`; original solutions are reproduced byte for byte with `-aos fixed`), then `python estudo_aos/run_aos_etapa7.py --full --workers 11` (about 4 h; resumable) and `python estudo_aos/analisar_etapa7.py`.
9b. **Studies 8-12** (all from `estudo_aos/`, WSL venv; each runner is resumable; the queue scripts `fila_estudos_9_10.sh`, `fila_12_11.sh` chain them): **Study 8** (positive control: planted useless operators, `src/operadores_fracos.py`): `python run_aos_etapa8.py --full` then `python analisar_etapa8.py`. **Study 9** (ALNS with stochastic destroy/repair operators, `src/alns.py`, `src/alns_loop.py`, validated by `python ../tests/test_alns.py`): `python run_aos_etapa9.py --tuning-grid`, `--selecionar`, `--eval`, `--vns` (VNS reference arms at the same budget), then `python analisar_etapa9.py`. **Study 10** (larger instances; needs `python gerar_instancias_sinteticas.py SYN_TUNE_L SYN_VAL_L` and `python gerar_solucoes_iniciais.py` first): `python run_aos_etapa10.py --tuning-static`, `--selecionar`, `--eval`, then `python analisar_etapa10.py`. **Study 11** (efficient SWAP*, `src/neighborhoods_fast.py`, validated by `python ../tests/test_swap_star_fast.py`): `python run_aos_etapa11.py --tuning-static`, `--selecionar`, `--eval74`, `--evalL`, then `python analisar_etapa11.py`. **Study 12** (static-mixture and fixed-repair control for the ALNS): `python run_aos_etapa12.py --tuning-static`, `--selecionar`, `--eval`, then `python analisar_etapa12.py`. The exploratory pilots (removal-size control `run_explor_q.py`, SYN_TUNE only) are not part of the confirmatory results.
9e. **Study 15** (prospective test of the pool dead-weight rule: nested pools P5-P8 of the TSP and the CVRPTW; `estudo_aos/run_aos_etapa15.py` (`--tuning-pools`, `--predizer`, `--eval-pools`), `analisar_etapa15.py`; pre-registration log entry 118, commit `7a6c7f6`; results entry 119).
9d. **Study 14** (generality: the CVRP with time windows, VND regime; `estudo_aos/run_aos_etapa14.py`, queue `fila_14.sh`, analysis `analisar_etapa14.py`; code in `src/tw_inst.py` (Solomon format, costs truncated to one decimal and stored in tenths), `src/tw_neighborhoods.py` (nine numba operators with O(1) time-window checks, validated by `tests/validate_tw_neighborhoods.py`), `src/solver_pyvrp_tw.py` (PyVRP references); instances: 56 Solomon instances + BKS in `instances/SOLOMON/` (downloaded from CVRPLIB, ids 380-435; `estudo_aos/verifica_solomon_bks.py` checks that our cost function reproduces the BKS 56/56) and 24 synthetic tuning instances `instances/SYN_TUNE_W` (`gerar_instancias_tw.py`); pre-registration log entry 116, commit `d66a191`; results entry 117; post hoc mode `--explor-perm`).
9c. **Study 13** (generality: the TSP; `estudo_aos/run_aos_etapa13.py`, queue `fila_13.sh`, analysis `analisar_etapa13.py`; TSP code in `src/tsp_inst.py`, `src/tsp_neighborhoods.py` (validated by `tests/validate_tsp_neighborhoods.py`), `src/tsp_alns.py`, `src/solver_lkh_tsp.py`; instances in `instances/TSPLIB/` (32 files downloaded from the official TSPLIB95 page, with the published optima) and `instances/SYN_TUNE_T`, `SYN_VAL_T`; pre-registration log entry 113, commit `27792ca`; results entries 114-115; `tests/test_tsp_regret_equiv.py` documents that regret-2 equals greedy insertion with one route; post hoc modes `--explor-ordem` and `--explor-perm`).
10. **Tables, figures, supplement and PDFs**: in `paper/ejor/` run `python gerar_tabelas.py` (all numbers of the Results section become `\num{key}` macros in `numeros.tex`; run it twice around the verification), `python gerar_figuras.py`, `python gerar_suplemento.py`, `python checar_macros.py`, then compile `main.tex` and `supplement.tex` (XeLaTeX or Tectonic; `ferramentas/tectonic` is a Linux binary used in WSL: `tectonic -X compile main.tex --outdir build`). See `paper/RASTREIO_NUMEROS.md` for the provenance of every number.
11. **Reference solver** (HGS-CVRP on the 74 instances): `python ../direcao1_local/rodar_hgs_74.py --regime matched|n01`.

Every runner appends one row per run to a CSV keyed by (arm, level, instance, seed) and skips rows already present,
so interrupted runs can be resumed by repeating the same command.

## Integrity controls

* Exact regression: each new version of the search loop reproduces the previous version's costs exactly under identical seeds (checked at every change; 48/48, 8/8 and 6/6 in the logged checks); the modified HGS-CVRP reproduces the original program's solution files byte for byte in `-aos fixed` mode (51/51); the five-operator arm of Study 11 reproduces the corresponding arm of Study 5 cost for cost (1 480/1 480).
* Operators validated against brute force (1 600 tests) with a mutation test.
* `verificacao_independente.py` must print `VERIFICACAO INDEPENDENTE: OK`.
* Git timestamps: commit `ef82cdc` precedes the Study 4 evaluation and `02e58d3` (extended by `b3d4add`) precedes all Study 6 data and `8a06b17` all Study 7 data; `43cccb0` precedes the Study 8 data, `d0347ac` the Study 9 and 10 data (the accounting of Study 9 was amended by `a61aab6`, entry 107, before any analysis of its data), `71a4553` the Study 11 data and `7977d5a` the Study 12 data, `27792ca` the Study 13 data, `d66a191` the Study 14 data; the pre-specification of Study 5 (entry 89) was in the research log before its data but was committed after its runs had started; Studies 1-3 are pre-specified only in the dated research log.
* `paper/ejor/checar_afirmacoes.py` re-derives the universal claims of the manuscript (no adaptive policy significantly better than its non-adaptive baseline) from the confirmatory tables; `paper/ejor/compilar.sh` compiles `main.tex` and `supplement.tex`; `paper/ejor/checar_macros.py` checks the manuscript against the EJOR author guide limits (highlights, keywords, abstract length) and that every `\num{key}` exists.

## License and third-party material

Instance files originate from CVRPLIB (<https://galgos.inf.puc-rio.br/cvrplib/>); sets A and B were proposed by Augerat et al. (1995), CMT by
Christofides et al. (1979) and X by Uchoa et al. (2017). HGS-CVRP is MIT-licensed. \[Choose and add a license for this repository before publication.\]
