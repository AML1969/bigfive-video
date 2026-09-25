"""The caveats of BS Profiler 3.0 (design 11) — the only source of these texts.

Every text is kept here word for word; the page, the characterization, the MBTI tab, the PDF and the journal take them
from this module and never write their own. C1 and C2 are the disclaimers of report.py (one string for the whole
copy). Texts with numbers are templates: `text(code, **values)` fills them, and the helpers `c6`, `c7`, `c13` take
the numbers from the frozen reference group (refnorms) and choose the Russian word forms, so the numbers in a text
always agree with the data.

Where each caveat goes (design 11):
- characterization: C11 (paragraph 4), C7 in short (paragraph 7), C14, C15, C12, when needed C13, C19, C20;
- tab «Тип MBTI» and section 2 of the PDF: C3, C4, C5, C6, C7, C9, C16; next to the letter strip C8, C18, C19;
  the language-model note C17;
- «Как получены оценки»: C13; page footer: C1, C2, C10, C3; tab «Данные»: C22; PDF «Как читать результаты»: C1, C2,
  C10, C11, C14, C15; no Big Five at all: C21 instead of the MBTI block.
"""
from __future__ import annotations

from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU

C1 = DISCLAIMER_RU
C2 = INTERVIEW_DISCLAIMER_RU

C3 = ("Тип MBTI здесь не измерен опросником: он пересчитан из четырёх шкал Big Five (экстраверсия → E–I, открытость "
      "опыту → S–N, доброжелательность → T–F, добросовестность → J–P) по опубликованным соответствиям шкал (McCrae, "
      "Costa, 1989). Система не определяет тип личности: она оценивает пять непрерывных шкал по видео и записывает "
      "четыре из них в нотации MBTI. Обратного перевода нет: по четырём буквам нельзя восстановить пять шкал, поэтому "
      "основой отчёта остаются шкалы Big Five.")
C4 = ("Оси E–I и S–N связаны со шкалами Big Five сильно (r ≈ 0.74 и 0.72), T–F и J–P — умеренно (r ≈ 0.44 и 0.49), "
      "поэтому буквы T/F и J/P ошибаются заметно чаще. Эти корреляции получены на самоотчётах по опросникам MBTI и "
      "NEO-PI, а не на оценках по видео: это соответствие шкал, а не точность нашей оценки.")
C5 = ("Ось «на границе» — значение попало в среднюю зону опорной группы, между 35-м и 65-м процентилем: небольшое "
      "изменение оценки поменяло бы букву. В записи с учётом границ такая ось обозначена X, рядом показана буква, "
      "которая получилась бы при строгом делении.")
C6_RU = ("Пороги для русской речи предварительные: граница между буквами — типичное (медианное) значение каждой системы "
         "среди {group}, обработанных системой до {date}. Буква E здесь означает «кажется экстравертнее, чем типичный "
         "ролик этой группы», а не принадлежность к экстравертам вообще. Группа мала и не представляет население; после "
         "калибровки на большей выборке буквы на осях рядом с границей могут измениться.")
C6_EN = ("Граница между буквами — медиана оценок наблюдателей в обучающей выборке First Impressions V2 (6000 роликов): "
         "буква E означает «кажется экстравертнее, чем половина людей этого датасета».")
C7_RU = ("Две системы обучены на разных данных и работают на разных шкалах, поэтому каждая переведена в буквы "
         "относительно своей опорной группы и показана отдельно, без усреднения. На {on_group} их оценки пока плохо "
         "согласуются между собой: строгие буквы двух систем совпали по оси E–I в {k_EI} {videos_EI} из {N}, по S–N — "
         "в {k_SN}, по T–F — в {k_TF}, по J–P — в {k_JP}. Поэтому основной считается OCEAN-AI; совпадение букв у двух "
         "систем — признак более надёжного вывода, расхождение — повод считать ось неустойчивой.")
C7_EN = ("OCEAN-AI и своя модель переведены в буквы по одной норме First Impressions V2. Совпадение букв у двух систем — "
         "признак более надёжного вывода, расхождение — повод считать ось неустойчивой.")
C8 = ("Тип по отрезкам считается по тем же порогам, что и тип за весь ролик, но оценка одного 20-секундного отрезка "
      "шумнее оценки всего ролика: смотрите, насколько устойчивы буквы, а не на отдельный отрезок. Если буква на оси "
      "часто меняется, вывод по этой оси неустойчив.")
C9 = ("Уверенность по оси — не вероятность того, что буква верна, а удалённость значения от границы: 0 — на самой "
      "границе, 1 — у края опорной группы.")
C10 = ("Эмоции, голос и мимика — сигналы отдельных моделей, обученных на англоязычных корпусах и фотографиях; это "
       "наблюдения о поведении на видео, а не диагноз. В оценку черт они не входят и сами по себе черт не доказывают.")
C11 = ("По первому впечатлению лучше всего считывается экстраверсия, хуже всего — эмоциональная устойчивость, поэтому "
       "выводы о нейротизме самые осторожные.")
C12 = ("Описания уровней показывают, как обычно выглядит поведение, создающее такое впечатление, а не пересказывают "
       "эпизоды этого ролика.")
C13 = ("В {k} {segments_k} из {n} система OCEAN-AI не дала оценки (чаще всего не распознаны речь или лицо); {these} в "
       "основные оценки, характеристику и тип и на графиках {shown}.")
C14 = ("Это описание того, как человек воспринимается по одной записи, — рабочая гипотеза для беседы, а не "
       "психологическое заключение и не основание для решений о человеке.")
C15 = "Ни характеристика, ни тип MBTI не являются оценкой пригодности человека к работе или учёбе."
C16 = "Названия типов условные и приведены для удобства чтения; в разных источниках они различаются."
C17 = ("Текст написан локальной языковой моделью по уже посчитанным числам и проверен программой на совпадение букв и "
       "чисел; тип она не пересчитывает, это вспомогательная формулировка, а не отдельный вывод.")
C18 = ("Оценки своей модели по отрезкам в заданиях, обработанных до версии 3.0, не сохранялись, поэтому её тип по "
       "отрезкам не показан.")
C19 = "Ролик короче 30 с оценивается целиком, одним отрезком, поэтому типа по ходу ролика нет."
C20 = ("Основная система OCEAN-AI не дала оценок по этому ролику, поэтому характеристика и тип построены по своей "
       "модели относительно её опорной группы.")
C21 = "Тип MBTI не рассчитан: в результате нет оценок Big Five."
C22 = "Раздел mbti для этого задания посчитан при показе версией 3.0 и в файл result.json не записан."

TEXTS = {
    "C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "C6-ru": C6_RU, "C6-en": C6_EN, "C7-ru": C7_RU, "C7-en": C7_EN,
    "C8": C8, "C9": C9, "C10": C10, "C11": C11, "C12": C12, "C13": C13, "C14": C14, "C15": C15, "C16": C16,
    "C17": C17, "C18": C18, "C19": C19, "C20": C20, "C21": C21, "C22": C22,
}
CODES = tuple(f"C{i}" for i in range(1, 23))


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Russian noun form for a count (the same rule as narrative2.plural_ru, which pulls in the chart libraries)."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def text(code: str, lang: str = "ru", **values) -> str:
    """The caveat `code` ('C1' … 'C22'); C6 and C7 exist for 'ru' and 'en'. Templates are filled with `values`
    (C6-ru: group, date; C7-ru: on_group, N, k_EI, videos_EI, k_SN, k_TF, k_JP; C13: k, segments_k, n, these, shown);
    the helpers c6, c7, c13 compute these values themselves."""
    if code in ("C6", "C7"):
        code = f"{code}-{'ru' if lang == 'ru' else 'en'}"
    t = TEXTS[code]
    return t.format(**values) if values else t


def c6(lang: str = "ru", reference: dict | None = None) -> str:
    """C6 for the reference group of the language (the frozen Russian group or the FIV2 norm)."""
    if lang != "ru":
        return C6_EN
    from . import refnorms
    ref = reference or refnorms.describe(refnorms.reference_for("oceanai", "ru"))
    return C6_RU.format(group=ref["group_ru"], date=refnorms.date_ru(ref.get("frozen_at")))


def on_group_ru(n: int) -> str:
    """'13 русских роликах' / '21 русском ролике' — the group in the prepositional case (after «на»)."""
    return f"{n} русском ролике" if _plural(n, "1", "2", "5") == "1" else f"{n} русских роликах"


def c7(lang: str = "ru", stats: dict | None = None) -> str:
    """C7 with the agreement of the two systems on the frozen Russian group (numbers from the norms file)."""
    if lang != "ru":
        return C7_EN
    from . import refnorms
    st = stats or refnorms.agreement_stats()
    n, same = int(st["n"]), st["letters_same"]
    return C7_RU.format(on_group=on_group_ru(n), N=n, k_EI=same["EI"],
                        videos_EI=_plural(same["EI"], "ролике", "роликах", "роликах"),
                        k_SN=same["SN"], k_TF=same["TF"], k_JP=same["JP"])


def c13(k: int, n: int) -> str:
    """«В 7 отрезках из 33 система OCEAN-AI не дала оценки …; эти отрезки не вошли … показаны пропусками.»"""
    one = int(k) == 1
    return C13.format(k=k, segments_k=_plural(k, "отрезке", "отрезках", "отрезках"), n=n,
                      these="этот отрезок не вошёл" if one else "эти отрезки не вошли",
                      shown="показан пропуском" if one else "показаны пропусками")
