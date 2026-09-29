"""The PDF report of BS Profiler 3.1, moving into this package step by step (refactoring plan of 3.1, stages 12–17).

So far: layout.py (the page geometry, the grey of the notes, the fonts), document.py (the page itself, class Report),
widgets.py (its card grid and score bars), fmt.py (the text helpers of the report), charts.py (the print charts),
mbti_section.py (the characterization block and the MBTI section), sections.py (the passport and the other sections
of the main part) and frames.py (what drove the score of AMLAI 1.0: the key frames, the modality chart, the words).
The plan of the report, the appendices and build_pdf are still bs3/pdf_report.py.
"""
