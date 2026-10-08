"""Build a local OA/Effusion review page and CSV template. No API calls."""

import argparse
import csv
import html
import json
from pathlib import Path

from extract import HERE, digest, read_jsonl, write_csv, write_text

FOCUS = ["Medial OA", "Lateral OA", "PF OA", "Effusion"]
FIELDS = ["StudyInstanceUID", "target", "llm_label", "report_status", "evidence",
          "evidence_strength", "score_reason",
          "uncertainty_reason", "review_verdict", "reviewed_label", "depth_evidence",
          "extent_evidence", "amount_evidence", "distension_evidence", "notes", "reviewer"]

PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>OA / Effusion 报告复核</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;color:#192b3b;background:#f1f5f9;margin:0}
header{position:sticky;top:0;background:#fff;padding:12px 24px;border-bottom:1px solid #b9c7d6;z-index:2}
main{max-width:1450px;margin:auto;padding:20px}button,select,input,textarea{font:inherit;padding:6px;border:1px solid #a7b6c5;border-radius:5px}
button{background:#174c6b;color:white;cursor:pointer;margin:4px}section{background:#fff;border-radius:10px;padding:18px;margin-bottom:22px}
.layout{display:grid;grid-template-columns:minmax(300px,1fr) minmax(400px,1.2fr);gap:20px}
.report{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7fafc;padding:15px;max-height:80vh;overflow:auto}
.card{border:1px solid #ccd7e0;padding:12px;margin-bottom:12px;border-radius:7px}.unknown{border-left:5px solid #b47319}
.quote{white-space:pre-wrap;background:#eef5fa;padding:8px;overflow-wrap:anywhere}textarea{box-sizing:border-box;width:100%;min-height:55px}
.uid{font-size:12px;overflow-wrap:anywhere}label{display:block;margin:6px 0}.reason{color:#8a4d0b}.hint{background:#fff7e6;padding:8px}
@media(max-width:850px){.layout{grid-template-columns:1fr}}summary{cursor:pointer;font-weight:bold}small{color:#465b6c}
</style>
<header><b>OA / Effusion 报告复核</b> <span id="progress"></span>
<button id="export">导出复核 CSV</button><label style="display:inline">显示 <select id="filter"><option value="all">全部</option><option value="unknown">含未知项的 study</option><option value="pending">有待复核项的 study</option></select></label>
<small id="save-status"></small></header>
<main><details open><summary>复核步骤与判定口径</summary>
<p>这是一份文本复核表，共 __COUNT__ 个 study、每份四项。先读全文，再核对证据、部位、程度、范围和软分数。分数是未校准的模型估计；旧目录显示的 0/1 保持原样，不会伪装成软标签。原始标签文件不会被修改。</p>
<p><b>OA：</b>分别确认内侧胫股、外侧胫股、髌股间室；记录超过 50% 深度和约 ≥1 cm 范围的证据。未给尺寸不等于无病变，也不必自动屏蔽软估计；用部分证据与较低权重表达。只说 OA 不能补造深度或范围。弥漫/广泛等描述不能伪装成毫米测量。</p>
<p><b>Effusion：</b>确认是关节液而非滑囊液，记录少量/中量/大量和扩张证据。少量或明确无积液可为 0；量不清保留未知。中大量但未明示扩张的情况标记“映射口径待确认”，避免病例之间一会儿严格、一会儿宽松。</p>
<p>复核判定：保留＝文本与当前规则一致；提取错误＝漏读、错译、否定/部位错误；映射口径待确认＝事实正确但阈值映射有争议；仍不确定＝报告无法补足证据。修订标签仅保存你的建议，不自动应用。</p>
<p>外语报告可以辅助翻译，但证据仍引用原文。页面不自动翻译。笔记尽量在每轮结束时导出 CSV；浏览器本地暂存可能受 file 页面限制。</p>
</details><div id="content"></div></main>
<script>
const DATA=__DATA__;const KEY='rsna-review-'+__KEY__;
let notes={};try{notes=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cardKey=(uid,t)=>uid+'|'+t;
const opts=(values,current)=>values.map(([v,n])=>`<option value="${esc(v)}" ${v===current?'selected':''}>${esc(n)}</option>`).join('');
function render(){
 const filter=document.getElementById('filter').value;
 document.getElementById('content').innerHTML=DATA.filter(s=>filter==='all'||(filter==='unknown'?s.items.some(i=>i.llm_label===''):s.items.some(i=>!notes[cardKey(s.uid,i.target)]?.review_verdict))).map(s=>
 `<section><h2>Study ${s.index}</h2><div class="uid">${esc(s.uid)}</div><div class="layout"><div><h3>报告原文</h3><div class="report">${esc(s.report)}</div></div><div>${s.items.map(i=>{
 const k=cardKey(s.uid,i.target),n=notes[k]||{};
 return `<div class="card ${i.llm_label===''?'unknown':''}" data-key="${esc(k)}"><b>${esc(i.target)}</b>　分数：${i.llm_label===''?'未知':i.llm_label}　状态：${esc(i.report_status)}
 <p>证据强度：${esc(i.evidence_strength)}　${esc(i.score_reason)}</p>
 <div class="quote">${esc(i.evidence.join('\n\n')||'无证据原句')}</div><p class="reason">${esc(i.uncertainty_reason)}</p>
 ${i.hint?`<div class="hint">${esc(i.hint)}</div>`:''}
 <label>复核判定 <select data-field="review_verdict">${opts([['','待复核'],['KEEP','保留'],['EXTRACTION_ERROR','提取错误'],['MAPPING_REVIEW','映射口径待确认'],['UNRESOLVED','仍不确定']],n.review_verdict||'')}</select></label>
 <label>建议软分数 <input data-field="reviewed_label" placeholder="0～1 或 UNKNOWN" value="${esc(n.reviewed_label||'')}"></label>
 ${(i.target==='Effusion'?[['amount_evidence','积液量原文'],['distension_evidence','关节扩张原文']]:[['depth_evidence','软骨深度/分级原文'],['extent_evidence','软骨范围原文']]).map(([f,t])=>`<label>${t}<textarea data-field="${f}">${esc(n[f]||'')}</textarea></label>`).join('')}
 <label>说明<textarea data-field="notes">${esc(n.notes||'')}</textarea></label><label>复核者<input data-field="reviewer" value="${esc(n.reviewer||'')}"></label></div>`}).join('')}</div></div></section>`).join('');
 progress();
}
function progress(){const done=DATA.flatMap(s=>s.items.map(i=>notes[cardKey(s.uid,i.target)]?.review_verdict)).filter(Boolean).length;document.getElementById('progress').textContent=`已记录 ${done}/${DATA.length*4} 项`;}
document.getElementById('content').addEventListener('input',e=>{const f=e.target.dataset.field;if(!f)return;const k=e.target.closest('[data-key]').dataset.key;notes[k]={...(notes[k]||{}),[f]:e.target.value};try{localStorage.setItem(KEY,JSON.stringify(notes));document.getElementById('save-status').textContent='已暂存；请导出 CSV'}catch(err){document.getElementById('save-status').textContent='无法暂存；请导出 CSV'}progress()});
document.getElementById('filter').addEventListener('change',render);
document.getElementById('export').addEventListener('click',()=>{
 const fields=__FIELDS__;const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
 const rows=DATA.flatMap(s=>s.items.map(i=>{const n=notes[cardKey(s.uid,i.target)]||{};return{StudyInstanceUID:s.uid,...i,evidence:JSON.stringify(i.evidence),...n}}));
 const csv=[fields.map(quote).join(','),...rows.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\n')+'\n';
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='oa_effusion_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});render();
</script></html>'''


def build_review(run):
    with (run / "selected_studies.csv").open(encoding="utf-8", newline="") as stream:
        studies = list(csv.DictReader(stream))
    results = {row["StudyInstanceUID"]: row for row in read_jsonl(run / "results.jsonl")}
    data, template = [], []
    for index, study in enumerate(studies, 1):
        uid = study["StudyInstanceUID"]
        if uid not in results:
            raise ValueError(f"Study has no successful extraction: {uid}")
        items = []
        for target in FOCUS:
            item = results[uid]["result"]["labels"][target]
            hint = ""
            if target != "Effusion" and item["report_status"] in {"PRESENT", "UNCERTAIN"}:
                hint = "重点核对：深度缺失、范围缺失还是部位不明？保留真实缺失；不要只为减少未知而改成阳性。"
            elif target == "Effusion" and item["label"] is not None and item["label"] > 0.5:
                hint = "重点核对：中大量是否明确？扩张有无独立原文？若只给量级，记录映射口径待确认。"
            items.append({"target": target, "llm_label": "" if item["label"] is None else item["label"],
                          "report_status": item["report_status"], "evidence": item["evidence"],
                          "evidence_strength": item.get("evidence_strength", "LEGACY_HARD"),
                          "score_reason": item.get("score_reason", "旧版本硬标签结果"),
                          "uncertainty_reason": item["uncertainty_reason"], "hint": hint})
            template.append({"StudyInstanceUID": uid, **{key: value for key, value in items[-1].items()
                                                         if key != "hint"}, "evidence": json.dumps(item["evidence"], ensure_ascii=False)})
        data.append({"index": index, "uid": uid, "report": study["Report"], "items": items})
    serial = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = PAGE.replace("__DATA__", serial).replace("__KEY__", json.dumps(digest(serial)))
    page = page.replace("__COUNT__", str(len(studies))).replace("__FIELDS__", json.dumps(FIELDS))
    write_text(run / "oa_effusion_review.html", page)
    template_path = run / "oa_effusion_review_template.csv"
    if not template_path.exists():
        write_csv(template_path, FIELDS, template)
    print(f"Review page: {run / 'oa_effusion_review.html'} ({len(template)} items)")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "output/pilot20_soft")
    build_review(parser.parse_args().run.resolve())
