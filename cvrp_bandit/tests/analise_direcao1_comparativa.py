#!/usr/bin/env python3
"""
analise_direcao1_comparativa.py
==================================
Consolida os 4 CSVs de resultado da Direção 1 (produzidos pelos scripts
de `direcao1_local/`, rodados na máquina do pesquisador -- ver diário,
Entradas 41-42) numa tabela comparativa única + testes de significância
pareados, seguindo a MESMA disciplina estatística usada em toda a
Direção 2 (`scipy.stats.wilcoxon` + `scipy.stats.ttest_rel`, pareado por
instância, usando a média dos 5 seeds quando existirem).

Lê (o que existir -- roda parcialmente se algum CSV ainda não chegou,
avisando quais faltam):
    ../results/tempos_hgucb_74.csv   (nosso HG-UCB, 5 seeds/instancia)
    ../results/ortools_74.csv        (OR-Tools, 5 seeds/instancia)
    ../results/lkh3_74.csv           (LKH3, 5 seeds/instancia)
    ../results/kool_avaliacao_74.csv (Kool fine-tunado, 1 decodificação gulosa/instancia)

IMPORTANTE sobre o Kool (ver achado #34, diário Entrada 42): o modelo não
consegue impor exatamente K rotas, então instâncias A/B reprovam quase
sempre no critério estrito de viabilidade. Por isso a comparação com o
Kool é feita em DUAS versões: (a) só nas instâncias `viavel_sem_restricao_k`
(universo maior, ignora a restrição de K -- mede "achou uma rota boa?"),
e (b) só nas `viavel_estrito` (universo menor, mesmo critério dos outros
solvers -- na prática deve sobrar quase só X/CMT). Reportar as duas evita
tanto inflar quanto invalidar artificialmente o resultado do Kool.

LKH3/OR-Tools: instâncias marcadas inviáveis (`viavel=False`) ou sem
solução (`custo` vazio) são EXCLUÍDAS do teste pareado numérico (não dá
pra tirar média de um campo vazio) mas contadas e reportadas à parte --
uma taxa de falha alta é, em si, um resultado a discutir, não algo para
esconder.

Uso: python analise_direcao1_comparativa.py
    (rodar de dentro de tests/, com os 4 CSVs em ../results/, ou apontar
    --results_dir para outro caminho, ex.: uma cópia recebida do
    pesquisador em /tmp/.../scratchpad/)
"""
import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

from scipy.stats import wilcoxon, ttest_rel

ROOT = Path(__file__).resolve().parent.parent
CONJUNTOS = ("A", "B", "X", "CMT")


def ler_csv(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def media_por_instancia(linhas, campo_instancia="instancia", campo_valor="gap_pct",
                        campo_conjunto="conjunto", so_viaveis=None):
    """Agrupa linhas (várias por instância, ex.: 5 seeds) e devolve
    {instancia: media}, {instancia: conjunto}, e a contagem de linhas
    excluídas por não terem valor numérico (inviável/sem solução).
    Se `so_viaveis` for o nome de uma coluna booleana, só usa linhas onde
    essa coluna é True/"True"."""
    soma, conta, conjunto_de = {}, {}, {}
    excluidas = 0
    for linha in linhas:
        if so_viaveis is not None and linha.get(so_viaveis) not in ("True", "true", "1", True):
            excluidas += 1
            continue
        valor = linha.get(campo_valor, "")
        if valor in ("", None):
            excluidas += 1
            continue
        nome = linha[campo_instancia]
        soma[nome] = soma.get(nome, 0.0) + float(valor)
        conta[nome] = conta.get(nome, 0) + 1
        conjunto_de[nome] = linha.get(campo_conjunto, "?")
    medias = {nome: soma[nome] / conta[nome] for nome in soma}
    return medias, conjunto_de, excluidas


def comparar_pareado(nome_a, medias_a, nome_b, medias_b, conjunto_de):
    """Compara duas séries {instancia: media_gap} nas instâncias em
    comum, geral e por conjunto -- mesmo padrão de sempre (Wilcoxon +
    t-test pareado, ver diário Entrada 25/30)."""
    comuns = sorted(set(medias_a) & set(medias_b))
    if len(comuns) < 2:
        print(f"  {nome_a} vs {nome_b}: menos de 2 instâncias em comum ({len(comuns)}) -- "
              f"não dá para testar significância.")
        return
    vals_a = [medias_a[n] for n in comuns]
    vals_b = [medias_b[n] for n in comuns]
    diff_medio = mean(vals_a) - mean(vals_b)
    vitorias_a = sum(1 for a, b in zip(vals_a, vals_b) if a < b)
    vitorias_b = sum(1 for a, b in zip(vals_a, vals_b) if b < a)
    empates = len(comuns) - vitorias_a - vitorias_b
    try:
        _, p_wilcoxon = wilcoxon(vals_a, vals_b)
    except ValueError as e:
        p_wilcoxon = float("nan")
        print(f"    (wilcoxon não pôde ser calculado: {e})")
    _, p_ttest = ttest_rel(vals_a, vals_b)
    print(f"  {nome_a} (media={mean(vals_a):.3f}%) vs {nome_b} (media={mean(vals_b):.3f}%)  "
          f"n={len(comuns)}  diff={diff_medio:+.4f}pp  "
          f"wilcoxon_p={p_wilcoxon:.4g}  ttest_p={p_ttest:.4g}  "
          f"vitorias({nome_a}={vitorias_a}, {nome_b}={vitorias_b}, empate={empates})")

    for c in CONJUNTOS:
        comuns_c = [n for n in comuns if conjunto_de.get(n) == c]
        if len(comuns_c) < 2:
            continue
        va = [medias_a[n] for n in comuns_c]
        vb = [medias_b[n] for n in comuns_c]
        print(f"      [{c:4s}] n={len(comuns_c):2d}  {nome_a}={mean(va):6.3f}%  "
              f"{nome_b}={mean(vb):6.3f}%  diff={mean(va)-mean(vb):+.4f}pp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    arquivos = {
        "hgucb": results_dir / "tempos_hgucb_74.csv",
        "ortools": results_dir / "ortools_74.csv",
        "lkh3": results_dir / "lkh3_74.csv",
        "kool": results_dir / "kool_avaliacao_74.csv",
    }
    dados = {nome: ler_csv(path) for nome, path in arquivos.items()}
    faltando = [nome for nome, d in dados.items() if d is None]
    if faltando:
        print(f"AVISO: ainda não encontrados (rodando só com o que existe): "
              f"{', '.join(arquivos[n].name for n in faltando)}\n")

    medias, conjuntos_de = {}, {}
    if dados["hgucb"]:
        medias["HG-UCB"], cj, _ = media_por_instancia(dados["hgucb"])
        conjuntos_de.update(cj)
    if dados["ortools"]:
        medias["OR-Tools"], cj, excl = media_por_instancia(dados["ortools"], so_viaveis="viavel")
        conjuntos_de.update(cj)
        if excl:
            print(f"OR-Tools: {excl} execuções excluídas (inviável ou sem valor de custo).")
    if dados["lkh3"]:
        medias["LKH3"], cj, excl = media_por_instancia(dados["lkh3"], so_viaveis="viavel")
        conjuntos_de.update(cj)
        if excl:
            print(f"LKH3: {excl} execuções excluídas (inviável ou sem solução dentro do orçamento).")
    if dados["kool"]:
        medias["Kool(sem-K)"], cj, _ = media_por_instancia(dados["kool"], so_viaveis="viavel_sem_restricao_k")
        conjuntos_de.update(cj)
        medias["Kool(estrito)"], cj2, _ = media_por_instancia(dados["kool"], so_viaveis="viavel_estrito")
        n_estrito = len(medias["Kool(estrito)"])
        print(f"Kool: {len(medias['Kool(sem-K)'])}/74 instâncias viáveis ignorando K, "
              f"{n_estrito}/74 viáveis no critério estrito (ver achado #34 -- esperado ser "
              f"pequeno, majoritariamente X/CMT).")

    if not medias:
        print("\nNenhum CSV de resultado encontrado ainda -- nada para comparar. "
              "Rode isto de novo depois que os resultados de direcao1_local/ existirem.")
        sys.exit(0)

    print(f"\n=== Gap médio por método (instâncias disponíveis, pode variar por método) ===")
    for nome, m in medias.items():
        print(f"  {nome:15s}  n={len(m):3d}  gap_medio={mean(m.values()):6.3f}%")

    if "HG-UCB" in medias:
        print(f"\n=== Comparações pareadas contra HG-UCB ===")
        for outro in ("OR-Tools", "LKH3", "Kool(sem-K)", "Kool(estrito)"):
            if outro in medias:
                comparar_pareado("HG-UCB", medias["HG-UCB"], outro, medias[outro], conjuntos_de)
    else:
        print("\n(tempos_hgucb_74.csv ainda não existe -- sem ele não dá pra comparar contra "
              "o HG-UCB, mas os outros métodos já podem ser comparados entre si acima.)")


if __name__ == "__main__":
    main()
