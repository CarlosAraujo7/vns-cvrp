#!/usr/bin/env python3
"""
test_tsp_regret_equiv.py
========================
Controle do Estudo 13 (escrito DEPOIS da avaliacao, ao notar que os reparos 'guloso' e 'regret-2' deram o mesmo gap com 15
casas decimais no ajuste): com UMA unica rota (max_rotas=1) o arrependimento-2 nao e definido (so ha uma opcao por cliente),
a chave do regret-2 degenera em (-inf, custo) e a escolha coincide com a da insercao gulosa. O teste confere, em N estados
parciais sorteados (instancias sinteticas, destruicoes aleatorias de tamanhos variados), que as duas reparacoes produzem
EXATAMENTE a mesma rota e que o 'barato' (ordem aleatoria) produz rotas diferentes. Como controle de sanidade, em estados com
3 rotas (max_rotas=3) o regret-2 difere do guloso em alguma fracao dos casos.

Uso: python tests/test_tsp_regret_equiv.py
"""
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from alns import Estado, REPARAR, destruir_aleatoria  # noqa: E402
from tsp_inst import ler_tsplib  # noqa: E402


def estado_tsp(inst, rng):
    ordem = list(range(1, inst.N))
    rng.shuffle(ordem)
    return Estado(inst, [ordem])


def main():
    rng = random.Random(7)
    iguais = difer_barato = total = 0
    for f in sorted((RAIZ / "instances" / "SYN_VAL_T").glob("*.tsp"))[:12]:
        inst = ler_tsplib(f)
        for _ in range(25):
            est0 = estado_tsp(inst, rng)
            q = rng.randint(3, max(4, inst.n_customers // 4))
            rem = destruir_aleatoria(est0, q, rng)
            res = []
            for j in (0, 1, 2):
                e = Estado(inst, est0.rotas)
                REPARAR[j][1](e, list(rem), random.Random(99), 1)
                res.append([list(r) for r in e.rotas])
            total += 1
            iguais += res[0] == res[1]
            difer_barato += res[2] != res[0]
    linhas = [f"TSP (1 rota): guloso == regret-2 em {iguais}/{total} estados; barato difere do guloso em {difer_barato}/{total}"]
    print(linhas[0])
    assert iguais == total, "regret-2 deveria coincidir com o guloso quando ha uma unica rota"
    # sanidade: com 3 rotas (max_rotas=3) o arrependimento-2 e definido e deve diferir do guloso em alguma fracao dos estados
    dif = n3 = 0
    for f in sorted((RAIZ / "instances" / "SYN_VAL_T").glob("*.tsp"))[:12]:
        inst = ler_tsplib(f)
        for _ in range(25):
            ordem = list(range(1, inst.N))
            rng.shuffle(ordem)
            t = len(ordem) // 3
            est0 = Estado(inst, [ordem[:t], ordem[t:2 * t], ordem[2 * t:]])
            q = rng.randint(3, max(4, inst.n_customers // 4))
            rem = destruir_aleatoria(est0, q, rng)
            res = []
            for j in (0, 1):
                e = Estado(inst, est0.rotas)
                REPARAR[j][1](e, list(rem), random.Random(99), 3)
                res.append([list(r) for r in e.rotas])
            n3 += 1
            dif += res[0] != res[1]
    linhas.append(f"3 rotas: regret-2 difere do guloso em {dif}/{n3} estados")
    print(linhas[-1])
    assert dif > 0, "com varias rotas o regret-2 deveria diferir do guloso em algum estado"
    (RAIZ / "results" / "aos_study" / "etapa13_regret_equiv.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("OK")


if __name__ == "__main__":
    main()
