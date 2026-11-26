#!/usr/bin/env python3
"""
run_aos_etapa30.py
==================
Estudo 30 da Direcao A (diario, Entrada 159 -- pre-registro): O QUE OS PESOS ADAPTATIVOS ACRESCENTAM A UMA FAIXA DE REMOCAO CALIBRADA? No ALNS do CVRP com 21 operadores (P21) os pesos de Ropke e Pisinger (RP) vencem a
uniforme por 0,30 pp nas 98 instancias novas do Estudo 26, e a uniforme com a faixa de remocao q multiplicada por c* = 0,6 (Uq, um parametro) recupera cerca de dois tercos disso (-0,20 pp). Duas leituras:
(i) o RP e um calibrador automatico do tamanho da remocao (e do uso dos operadores) e nada mais; (ii) ele acrescenta algo alem disso. Aqui a escala de q tambem e dada ao RP, e a dinamica do RP e separada da sua
alocacao media por um braco de REPETICAO estatica:
  RPq    RP (rho 0,1; segmento 100) com a faixa de remocao multiplicada por c*_RP, escolhido em {0,25; 0,4; 0,6; 1} em SYN_TUNE (seeds 3000-3002, pelo menor gap medio; empate: o maior c; se c* = 1, RPq = RP)
  SRPq   mistura estatica cujos pesos sao o uso medio do RPq nas 72 execucoes de VALIDACAO (SYN_TUNE, seeds 3003-3005), com a mesma escala de q
Os bracos U, Uq, RP (e S65, HG) sao os do Estudo 26 (etapa26_eval.csv), reutilizados sem nova execucao nas MESMAS 98 instancias e seeds (X_FRESH 38 + SYN_REP 60; seeds 1000 e 1001).
Modos: --regressao | --calibrar | --selecionar | --validar | --eval (retomaveis por chave). Uso (WSL): python3 -u run_aos_etapa30.py --calibrar --workers 11
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa21 as r21  # noqa: E402
import run_aos_etapa23 as r23  # noqa: E402
import run_aos_etapa26 as r26  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402

RES = r23.RES
SEEDS_TUNE = r23.SEEDS_TUNE                       # 3000-3002
SEEDS_VALID = [3003, 3004, 3005]
SEMENTES_AVAL = r26.SEMENTES_AVAL                 # 1000
SEEDS30 = r26.SEEDS26                             # 2 seeds por instancia
ESCALAS = r23.ESCALAS                             # (0,25; 0,4; 0,6; 1)
PADRAO_RP = dict(r21.PADRAO_RP)                   # rho 0,1; segmento 100


def spec_rpq(nome, c, config_id="e30"):
    return r23.com_id(r23.spec23(nome, "RP", params=dict(PADRAO_RP), q_escala=c, config_id=config_id), "RPq")


def tarefas_tune(sp, seeds):
    return [(sp, 1, "SYN_TUNE", n, s) for n in sorted(r4.nomes("SYN_TUNE"), reverse=True) for s in seeds]


# ------------------------------------------------------------------------------------------------ regressao exata
def modo_regressao(workers):
    """Reproduz EXATAMENTE (custo, iteracoes, avaliacoes, usos, escores) 12 execucoes de etapa26_eval.csv (3 de U, Uq e RP em instancias sinteticas de 100 clientes) com as especificacoes congeladas do Estudo 26
    e 3 execucoes do RP com o RPq de escala 1 (que deve ser identico ao RP)."""
    import pandas as pd
    rs = random.Random(30)
    e26 = pd.read_csv(RES / "etapa26_eval.csv")
    sp26 = r26.specs26()
    casos = []
    for a in ("U", "Uq", "RP"):
        d = e26[(e26.braco == f"s26_{a}") & (e26.conjunto == "SYN_REP") & (e26.n_customers == 100)]
        for i in rs.sample(sorted(d.instancia.unique()), 3):
            ref = d[d.instancia == i]
            casos.append((a, sp26[a], ref.iloc[rs.randrange(len(ref))]))
    d = e26[(e26.braco == "s26_RP") & (e26.conjunto == "SYN_REP") & (e26.n_customers == 100)]
    for i in rs.sample(sorted(d.instancia.unique()), 3):
        ref = d[d.instancia == i]
        casos.append(("RPq1", spec_rpq("reg_RPq1", 1.0, config_id="e26"), ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (_a, s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.starmap(r23.executar_23, tar)
    for (a, _s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (a, nv["instancia"], lin.instancia)
        for campo in r21.CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (a, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (a, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes do Estudo 26 reproduzidas; o RPq de escala 1 e identico ao RP)", flush=True)


# ------------------------------------------------------------------------------------------------ calibracao da escala de q do RP, selecao, validacao
def modo_calibrar(workers):
    out = RES / "etapa30_calibracao_q.csv"
    nome = "cal30_RPq"
    tar = []
    for c in ESCALAS:
        tar += tarefas_tune(spec_rpq(nome, c, config_id=f"c{c}"), SEEDS_TUNE)
    print(f"calibracao da escala de q do RP: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, out, pool)
    print("calibracao concluida", flush=True)


def modo_selecionar():
    import pandas as pd
    d = pd.read_csv(RES / "etapa30_calibracao_q.csv")
    d = d[d.viavel]
    n_ref = len(r4.nomes("SYN_TUNE")) * len(SEEDS_TUNE)
    cont = d.groupby("config_id").gap_pct.count()
    assert all(int(cont[f"c{c}"]) == n_ref for c in ESCALAS), cont.to_dict()
    g = d.groupby("config_id").gap_pct.mean()
    melhor = min(ESCALAS, key=lambda c: (round(float(g[f"c{c}"]), 12), -c))                 # empate: o maior c (o mais proximo do RP padrao)
    esc = {"c_estrela": melhor, "gap_ajuste_RPq": {str(c): float(g[f"c{c}"]) for c in ESCALAS}, "c_estrela_Uq_estudo23": json.loads((RES / "etapa23_selecao.json").read_text())["c_estrela"]}
    (RES / "etapa30_selecao.json").write_text(json.dumps(esc, indent=1))
    print(g.round(4).to_string())
    print(f"c* do RP = {melhor} ({'o RPq e o RP padrao' if melhor == 1.0 else 'RPq novo'}); c* da uniforme (Estudo 23) = {esc['c_estrela_Uq_estudo23']}", flush=True)


def selecao():
    return json.loads((RES / "etapa30_selecao.json").read_text())


def modo_validar(workers):
    """72 execucoes do RPq (escala c*) nas instancias de ajuste com seeds novas; o uso medio vira os pesos do SRPq."""
    import pandas as pd
    esc = selecao()
    sp = spec_rpq("val30_RPq", esc["c_estrela"], config_id="e30")
    tar = tarefas_tune(sp, SEEDS_VALID)
    print(f"validacao do RPq (escala {esc['c_estrela']}): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, RES / "etapa30_validacao.csv", pool)
    v = pd.read_csv(RES / "etapa30_validacao.csv")
    v = v[v.viavel]
    assert len(v) == 72, len(v)
    out = []
    for col in ("uso_d", "uso_r"):
        m = pd.DataFrame(v[col].str.split("|").apply(lambda s: [float(x) for x in s]).tolist()).mean()
        out.append([round(float(x), 4) for x in m / m.sum()])
    (RES / "etapa30_srpq_pesos.json").write_text(json.dumps({"pesos_d": out[0], "pesos_r": out[1], "c_estrela": esc["c_estrela"]}, indent=1))
    print(f"gap medio da validacao do RPq: {v.gap_pct.mean():.4f}; pesos de SRPq: destruicao {out[0]}; reparo {out[1]}", flush=True)


# ------------------------------------------------------------------------------------------------ avaliacao
def specs_eval():
    esc = selecao()
    w = json.loads((RES / "etapa30_srpq_pesos.json").read_text())
    sp = {}
    if esc["c_estrela"] != 1.0:
        sp["RPq"] = spec_rpq("s30_RPq", esc["c_estrela"])
    s = r23.spec23("s30_SRPq", "static", q_escala=esc["c_estrela"], config_id="e30")
    s["pesos_d"], s["pesos_r"] = w["pesos_d"], w["pesos_r"]
    sp["SRPq"] = r23.com_id(s, "SRPq")
    return sp


def modo_eval(workers):
    sp = specs_eval()
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, _nc in r26.instancias26() for s in sp.values() for sd in range(SEEDS30)]
    print(f"eval Estudo 30 (P21, 98 instancias novas): {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, RES / "etapa30_eval.csv", pool)
    print("eval Estudo 30 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("regressao", "calibrar", "selecionar", "validar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    if a.calibrar:
        return modo_calibrar(a.workers)
    if a.selecionar:
        return modo_selecionar()
    if a.validar:
        return modo_validar(a.workers)
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
