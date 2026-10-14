"""Local, escaped HTML evidence report; no remote assets or scripts."""
from html import escape
import json
from pathlib import Path


def write_report(report,output):
    output=Path(output)
    body=['<!doctype html><meta charset="utf-8"><title>Visual policy report</title>',
          '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#fafafa}pre{white-space:pre-wrap;background:#eee;padding:16px}img{max-width:460px}video{width:800px;max-width:100%}section{border-top:1px solid #bbb;padding:20px 0}</style>',
          '<h1>Visual policy simulation</h1>',
          '<p>'+escape(report['task'])+'</p><p>Status: <b>'+escape(report['status'])+'</b></p>',
          '<p>Simulation evidence only. Model statements and commanded movement are not success verification.</p>']
    if (output/'simulation.mp4').exists():body.append('<video controls src="simulation.mp4"></video>')
    for entry in report.get('history',[]):
        body.append('<section><h2>Step '+str(entry['step'])+'</h2>')
        for key in ('before_image','after_image'):
            if key in entry:
                body.append('<img alt="'+key+'" src="'+escape(Path(entry[key]).name,quote=True)+'">')
        body.append('<pre>'+escape(json.dumps(entry,ensure_ascii=False,indent=2))+'</pre></section>')
    body.append('<pre>'+escape(json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False,indent=2))+'</pre>')
    (output/'report.html').write_text('\n'.join(body),encoding='utf-8')
