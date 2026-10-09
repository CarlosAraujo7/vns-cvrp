# Does adaptive operator selection help? Replication package

Code, instances, raw results and analysis scripts for the study *"Does adaptive operator selection help? A controlled,
pre-specified study of bandit-guided variable neighborhood search for the capacitated vehicle routing problem"*
(manuscript in `paper/ejor/`). The research log with the dated pre-specification of every study is
`../diario_pesquisa.md` (Portuguese; entries 64, 71, 77, 83, 89 and 91 are the pre-registrations of Studies 1-6).

## Layout

| Path | Contents |
|---|---|
| `src/` | Core code: instance reader, construction, neighborhoods (`neighborhoods.py`: 5 original operators; `neighborhoods_ext.py`: 5 added operators), the reproduced HG-UCB bandit (`bandit.py`), search loops (`vns.py`, `aos_vnd.py`), classical AOS policies (`aos_policies.py`) and the additional policies of Study 6 (`aos_policies_ext.py`: EXP3, Thompson sampling, sliding-window UCB, adaptive pursuit, efficiency UCB), distance-evaluation counter (`instancia_contadora.py`), stored initial solutions loader (`solucao_inicial.py`) |
| `estudo_aos/` | Runners and analyses of Studies 1-6 (`run_aos_*.py`, `analisar_*.py`), robustness analysis, independent verification, synthetic-instance generator, initial-solution generator, structure diagnostic (`diag_estrutura.py`), Study 4 and 6 pipelines |
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
9. **Tables, figures, supplement and PDFs**: in `paper/ejor/` run `python gerar_tabelas.py` (all numbers of the Results section become `\num{key}` macros in `numeros.tex`; run it twice around the verification), `python gerar_figuras.py`, `python gerar_suplemento.py`, `python checar_macros.py`, then compile `main.tex` and `supplement.tex` (XeLaTeX or Tectonic; `ferramentas/tectonic` is a Linux binary used in WSL: `tectonic -X compile main.tex --outdir build`). See `paper/RASTREIO_NUMEROS.md` for the provenance of every number.
10. **Reference solver** (HGS-CVRP on the 74 instances): `python ../direcao1_local/rodar_hgs_74.py --regime matched|n01`.

Every runner appends one row per run to a CSV keyed by (arm, level, instance, seed) and skips rows already present,
so interrupted runs can be resumed by repeating the same command.

## Integrity controls

* Exact regression: each new version of the search loop reproduces the previous version's costs exactly under identical seeds (checked at every change; 48/48, 8/8 and 6/6 in the logged checks).
* Operators validated against brute force (1 600 tests) with a mutation test.
* `verificacao_independente.py` must print `VERIFICACAO INDEPENDENTE: OK`.
* Git timestamps: commit `ef82cdc` precedes the Study 4 evaluation and `02e58d3` precedes all Study 6 data; the pre-specification of Study 5 (entry 89) was in the research log before its data but was committed after its runs had started; Studies 1-3 are pre-specified only in the dated research log.
* `paper/ejor/checar_afirmacoes.py` re-derives the universal claims of the manuscript (no adaptive policy significantly better than its non-adaptive baseline) from the confirmatory tables; `paper/ejor/compilar.sh` compiles `main.tex` and `supplement.tex`; `paper/ejor/checar_macros.py` checks the manuscript against the EJOR author guide limits (highlights, keywords, abstract length) and that every `\num{key}` exists.

## License and third-party material

Instance files originate from CVRPLIB (<https://galgos.inf.puc-rio.br/cvrplib/>); sets A and B were proposed by Augerat et al. (1995), CMT by
Christofides et al. (1979) and X by Uchoa et al. (2017). HGS-CVRP is MIT-licensed. \[Choose and add a license for this repository before publication.\]
