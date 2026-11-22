#!/usr/bin/env python3
"""
run_aos_etapa29.py
==================
Estudo 29 da Direcao A (diario, Entrada 156 -- pre-registro): OS BRACOS ADAPTATIVOS AJUSTADOS NO ALNS DO CVRPTW COM 21 OPERADORES. No Estudo 28 a mistura estatica ajustada com 33 avaliacoes
(S33) venceu os pesos adaptativos de Ropke e Pisinger (RP, hiperparametros PADRAO: rho 0,1, segmento 100) por 0,47 pp e a uniforme por 0,65 pp no conjunto de 21 operadores do CVRPTW, mas o RP
nao foi ajustado nesse conjunto e o HG-UCB estava ajustado para o conjunto de 7 operadores (ameacas 3 e 4 da auditoria da Entrada 152). Aqui os dois bracos adaptativos recebem o MESMO esforco de
ajuste da mistura (33 avaliacoes de 72 execucoes nas 24 instancias de SYN_TUNE_W, seeds 3000-3002), por busca aleatoria de 2 hiperparametros cada:
  RPt   RP com (rho em [0,02; 0,5] log-uniforme, segmento em {25, 50, 100, 200, 400}): a configuracao padrao r00 + 32 sorteios (o espaco dos Estudos 18 e 21)
  HGt   HG-UCB com (C em [0,3; 6,0] log-uniforme, alpha em [0,03; 3,0] log-uniforme): a configuracao do Estudo 28 (C 2,0; alpha 0,5; r00) + 32 sorteios
A configuracao escolhida de cada familia e a de menor gap medio de ajuste (se a padrao for a melhor, o braco e a padrao e a coluna do Estudo 28 e copiada). Avaliacao: as 56 instancias de Solomon, seeds
1000-1004, as MESMAS do Estudo 28, cujos bracos (U, RP, HG, S33, S65) sao REUTILIZADOS sem nova execucao (etapa28_eval.csv). Validacao (diagnostico): seeds 3003-3005 nas instancias de ajuste.
Modos: --regressao | --ajuste {RP,HG} | --selecionar | --validar | --eval (retomaveis por chave). Uso (WSL): python3 -u run_aos_etapa29.py --ajuste RP --workers 11
O executor e o do Estudo 28 (`run_aos_etapa28.executar_28`): `--regressao` reproduz EXATAMENTE 12 execucoes de etapa28_eval.csv (U, RP, HG, S33) antes de qualquer dado novo.
"""
import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa18 as r18  # noqa: E402
import run_aos_etapa28 as r28  # noqa: E402

RES = r28.RES
SEEDS_TUNE = r28.SEEDS_TUNE                       # 3000-3002
SEEDS_VALID = r28.SEEDS_VALID                     # 3003-3005
SEMENTES_AVAL = r28.SEMENTES_AVAL                 # 1000
SEEDS_AVAL = r28.SEEDS_AVAL                       # 5
N_AVAL = r18.N_AVAL                               # 33 avaliacoes de ajuste (a mesma da mistura S33)
SEMENTE_RS = {"RP": 20262911, "HG": 20262912}
FAMILIAS = ("RP", "HG")
POL = {"RP": "RP", "HG": "hgucb"}
NOME_TUNE = {"RP": "tune29_RP", "HG": "tune29_HG"}
NOME_EVAL = {"RP": "s29_RPt", "HG": "s29_HGt"}


def padrao(fam):
    """Configuracao padrao (r00): a do RP do Estudo 28 e o HG-UCB do Estudo 28 (o ajustado para P7 no Estudo 16)."""
    return dict(r28.PADRAO_RP) if fam == "RP" else dict(r28.esc16()["hgucb"]["params"])


def espaco(fam, rng):
    """Um sorteio do espaco de hiperparametros da familia (busca aleatoria; valores arredondados a 4 algarismos significativos)."""
    if fam == "RP":
        return r18.espaco("RP18", rng)
    r4 = lambda x: float(f"{x:.4g}")  # noqa: E731
    lu = lambda a, b: r4(math.exp(rng.uniform(math.log(a), math.log(b))))  # noqa: E731
    return {"C": lu(0.3, 6.0), "alpha": lu(0.03, 3.0)}


def configs_rs(fam):
    """33 configuracoes: a padrao (r00) e 32 sorteios: lista de (config_id, params)."""
    rng = random.Random(SEMENTE_RS[fam])
    cfgs = [("r00", padrao(fam))]
    for i in range(1, N_AVAL):
        cfgs.append((f"r{i:02d}", espaco(fam, rng)))
    return cfgs


def csv_ajuste(fam):
    return RES / f"etapa29_ajuste_{fam}.csv"


def modo_ajuste(fam, workers):
    out = csv_ajuste(fam)
    tar = []
    for cid, params in configs_rs(fam):
        tar += r28.tarefas_ajuste(r28.spec28("P21", NOME_TUNE[fam], POL[fam], params=params, config_id=cid), SEEDS_TUNE)
    print(f"ajuste {fam}: {len(tar)} execucoes ({N_AVAL} configuracoes)", flush=True)
    with mp.Pool(workers) as mpool:
        r28.rodar28(tar, out, mpool)
    print(f"ajuste {fam} concluido", flush=True)


def modo_selecionar():
    esc = {}
    n_ref = len(r28.nomes(r28.TUNE_CONJ)) * len(SEEDS_TUNE)
    for fam in FAMILIAS:
        lst = r18.melhor_por_config(csv_ajuste(fam), NOME_TUNE[fam])
        assert len(lst) == N_AVAL and all(x[3] == n_ref for x in lst), (fam, len(lst), [x[3] for x in lst if x[3] != n_ref])
        pad = [x for x in lst if x[0] == "r00"][0]
        t = lst[0] if lst[0][2] < pad[2] - 1e-12 else pad                      # empate ou pior que o padrao -> padrao
        params = {k: v for k, v in json.loads(t[1]).items() if k != "pool"}
        esc[fam] = {"mudou": t[0] != "r00", "params": params, "gap_ajuste": t[2], "gap_padrao": pad[2], "config_id": t[0], "n_configs": len(lst),
                    "melhores": [{"config_id": c, "params": {k: v for k, v in json.loads(p).items() if k != "pool"}, "gap_ajuste": g} for c, p, g, _n in lst[:5]]}
        print(f"{fam}: padrao {pad[2]:.4f}  escolhida {t[2]:.4f} (config {t[0]}, {params}; {'MUDOU' if esc[fam]['mudou'] else 'padrao'})  pior config {lst[-1][2]:.4f}")
    (RES / "etapa29_ajuste_escolhidas.json").write_text(json.dumps(esc, indent=1))


def specs_eval():
    esc = json.loads((RES / "etapa29_ajuste_escolhidas.json").read_text())
    return {NOME_EVAL[f]: r28.spec28("P21", NOME_EVAL[f], POL[f], params=esc[f]["params"]) for f in FAMILIAS if esc[f]["mudou"]}


def modo_validar(workers):
    """Diagnostico (seeds 3003-3005, instancias de ajuste): RPt e HGt (inclusive se iguais a padrao); os bracos do Estudo 28 ja tem a validacao em etapa28_validacao.csv."""
    esc = json.loads((RES / "etapa29_ajuste_escolhidas.json").read_text())
    sp = {NOME_EVAL[f]: r28.spec28("P21", NOME_EVAL[f], POL[f], params=esc[f]["params"]) for f in FAMILIAS}
    tar = [t for s in sp.values() for t in r28.tarefas_ajuste(s, SEEDS_VALID)]
    print(f"validacao (instancias de ajuste, seeds novas): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as mpool:
        r28.rodar28(tar, RES / "etapa29_validacao.csv", mpool)
    import pandas as pd
    d = pd.read_csv(RES / "etapa29_validacao.csv")
    print(d[d.viavel].groupby("braco").gap_pct.mean().sort_values().round(4).to_string())


def modo_eval(workers):
    sp = specs_eval()
    if not sp:
        print("nenhum braco novo (a configuracao padrao venceu nas duas familias): nada a avaliar", flush=True)
        return
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, _nc in r28.instancias_aval() for s in sp.values() for sd in range(SEEDS_AVAL)]
    print(f"eval Estudo 29 (P21 do CVRPTW): {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as mpool:
        r28.rodar28(tar, RES / "etapa29_eval.csv", mpool)
    print("eval Estudo 29 concluido", flush=True)


def modo_regressao(workers):
    """Reproduz EXATAMENTE (custo, iteracoes, avaliacoes, usos e escores) 12 execucoes de etapa28_eval.csv (3 de cada braco U, RP, HG, S33) com o executor do Estudo 28."""
    import pandas as pd
    rs = random.Random(29)
    e28 = pd.read_csv(RES / "etapa28_eval.csv")
    sp28 = r28.specs_eval()
    casos = []
    for b in ("s28_U", "s28_RP", "s28_HG", "s28_S33"):
        d = e28[e28.braco == b]
        for i in rs.sample(sorted(d.instancia.unique()), 3):
            ref = d[d.instancia == i]
            casos.append((b, sp28[b], ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (_b, s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as mpool:
        novas = mpool.map(r28._worker28, tar)
    for (b, _s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (b, nv["instancia"], lin.instancia)
        for campo in r28.CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (b, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (b, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes do Estudo 28 reproduzidas)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regressao", action="store_true")
    g.add_argument("--ajuste", choices=FAMILIAS)
    g.add_argument("--selecionar", action="store_true")
    g.add_argument("--validar", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    if a.ajuste:
        return modo_ajuste(a.ajuste, a.workers)
    if a.selecionar:
        return modo_selecionar()
    if a.validar:
        return modo_validar(a.workers)
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
