from __future__ import annotations

from .parameter_resolution import (
    ParameterResolution,
    apply_human_decision,
    known_parameters,
)


def review_parameter_resolution(
    resolution: ParameterResolution,
) -> ParameterResolution | None:
    """
    Interface simples de terminal para revisão humana
    de parâmetros sugeridos pela LLM.

    Retorna:
        ParameterResolution -> decisão aprovada
        None                -> permanece pendente
    """

    suggestion = resolution.llm_suggestion

    print("\n" + "=" * 70)
    print("REVISÃO HUMANA DE PARÂMETRO")
    print("=" * 70)

    print(f"\nOriginal: {resolution.original}")

    if suggestion is not None:
        print(f"Sugestão: {suggestion.suggested_canonical}")
        print(f"Unidade: {suggestion.suggested_unit}")
        print(f"Confiança: {suggestion.confidence:.2f}")
        print(f"Motivo: {suggestion.reason}")
    else:
        print("A LLM não encontrou correspondência.")

    print(
        "\n"
        "[1] Aceitar sugestão da LLM\n"
        "[2] Corrigir para outro parâmetro canônico\n"
        "[3] Cadastrar como novo parâmetro\n"
        "[4] Manter pendente"
    )

    choice = input("\nEscolha: ").strip()

    # ========================================================
    # 1. ACEITAR SUGESTÃO
    # ========================================================

    if choice == "1":

        if (
            suggestion is None
            or suggestion.suggested_canonical is None
        ):
            print("Não existe sugestão para aceitar.")
            return None

        return apply_human_decision(
            original=resolution.original,
            canonical=suggestion.suggested_canonical,
            unit=suggestion.suggested_unit,
        )

    # ========================================================
    # 2. CORRIGIR PARA CANÔNICO EXISTENTE
    # ========================================================

    if choice == "2":

        known = known_parameters()

        print("\nParâmetros canônicos conhecidos:\n")

        for index, item in enumerate(known, start=1):
            print(
                f"[{index}] "
                f"{item['canonical']} "
                f"({item['unit'] or 'sem unidade'})"
            )

        selected = input(
            "\nNúmero do parâmetro correto: "
        ).strip()

        try:
            selected_index = int(selected) - 1
            item = known[selected_index]

        except (ValueError, IndexError):
            print("Seleção inválida.")
            return None

        return apply_human_decision(
            original=resolution.original,
            canonical=item["canonical"],
            unit=item["unit"],
        )

    # ========================================================
    # 3. NOVO PARÂMETRO
    # ========================================================

    if choice == "3":

        canonical = input(
            "Nome canônico: "
        ).strip()

        if not canonical:
            print("Nome canônico não pode ser vazio.")
            return None

        unit = input(
            "Unidade canônica (Enter se desconhecida): "
        ).strip()

        if not unit:
            unit = None

        return apply_human_decision(
            original=resolution.original,
            canonical=canonical,
            unit=unit,
        )

    # ========================================================
    # 4. MANTER PENDENTE
    # ========================================================

    if choice == "4":
        return None

    print("Opção inválida.")
    return None

def review_pending_parameters(
    pending: list[ParameterResolution],
) -> list[ParameterResolution]:

    approved = []

    for resolution in pending:

        decision = review_parameter_resolution(
            resolution
        )

        if decision is not None:
            approved.append(decision)

    return approved