"""The web UI of BS Profiler 3.1 (refactoring plan of 3.1, stages 18–19).

The pieces so far: charts.py (the interactive plotly figures of the page), plotframe.py (embeds a figure as an
<iframe srcdoc> with the theme switch and the page font), parts.py (the HTML helpers — tables, score bars, the
modality table) and mbti_html.py (the tab «Тип MBTI» and the short emotion paragraph). The Gradio app itself is still
bs3/webapp.py until stage 19.
"""
