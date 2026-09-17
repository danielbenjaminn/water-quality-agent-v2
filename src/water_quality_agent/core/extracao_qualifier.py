import numpy as np
from typing import Any, Tuple, List


# Lista global para armazenar os índices com erro e seus valores
indices_com_erro: List[Tuple[int, Any]] = []


def is_valid_number(num: Any) -> bool:
    """Determina se um valor pode ser convertido para float."""
    try:
        float(num)
        return True
    except (ValueError, TypeError):
        return False


def convert_to_float(number: Any) -> float:
    """Converte um valor para float ou retorna NaN quando vazio."""
    if number is not None and len(str(number)) > 0:
        return float(number)

    return np.nan


def extrair_qualif(res: Any, index: int) -> Tuple[Any, float]:
    """
    Separa o qualifier do valor numérico de um resultado analítico.

    Exemplos:
        "7.2"            -> (nan, 7.2)
        "<0.01"          -> ("<", 0.01)
        "Presente"       -> ("Presente", 1)
        "Ausente"        -> ("Ausente", 0)
        "Não Detectável" -> ("Não Detectável", nan)
    """

    try:
        # Trata None, NaN e string vazia
        if res is None:
            return (np.nan, np.nan)

        if isinstance(res, float) and np.isnan(res):
            return (np.nan, np.nan)

        res = str(res).strip()

        if not res:
            return (np.nan, np.nan)

        # Resultado puramente numérico
        if is_valid_number(res):
            aux = (np.nan, float(res))

        # Resultado com qualifier < ou >
        elif res[0] in ["<", ">"]:

            if is_valid_number(res[1:]):
                aux = (
                    res[0],
                    convert_to_float(res[1:])
                )

            elif res[1:] == "LQ":
                aux = (res, np.nan)

            else:
                aux = (
                    res[0],
                    convert_to_float(res[1:])
                )

        # Resultados qualitativos
        elif res in ["Ausente", "Virt. Ausente"]:
            aux = (res, 0)

        elif res in ["Presente", "Virt. Presente"]:
            aux = (res, 1)

        elif res in [
            "N.A.",
            "Não Detectável",
            "Não Objetável",
            "Desligado",
            "Repartida",
            "Partindo",
            "Não Amostrado",
            "Fora do Lim. Detec.",
            "Ponto Seco",
            "Lâm. d´água insuficiente",
            "Amostra Extraviada",
            "Em Manutenção",
        ]:
            aux = (res, np.nan)

        # Última tentativa: interpretar como número
        else:
            aux = (
                np.nan,
                convert_to_float(res)
            )

    except (ValueError, TypeError):
        indices_com_erro.append((index, res))
        aux = (np.nan, np.nan)

    return aux