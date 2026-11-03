"""Readable, evidence-linked scorer output; never converts a score into acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from contracts import require
from evaluator_common import write_json
from source_lock import sha


def numeric(value):
    if isinstance(value,dict):
        require(set(value)=={'type','real','imag'} and value['type']=='complex', 'Unknown metric scalar encoding')
        return complex(value['real'],value['imag'])
    if isinstance(value,list):
        return [numeric(x) for x in value]
    return value


def metric_values(result):
    if 'metrics' in result: return result['metrics']
    return {name: values['candidate_'+result['arm']]['mean']
            for name,values in result['metric_summary'].items()}


def identity(result):
    inputs = [{k:r[k] for k in ('source_id','native_sha256','text_sha256','audio_sha256') if k in r}
              for r in result['inputs']]
    reference = result.get('reference_scope')
    if isinstance(reference,dict): reference = reference['files']
    beat_protocol = None
    if result['dataset'] == 'BEAT2':
        beat_protocol = dict(profile=result.get('beat_profile', 'public2025'),
                             receipt=result.get('profile_receipt_sha256'),
                             support=[{k:r[k] for k in ('frames','old_loader_frames','bc_frames') if k in r}
                                      for r in result['inputs']])
    value = [result['dataset'],result['split'],inputs,reference,result.get('feature_protocol'),beat_protocol]
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--result',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True); args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows, originals = [], {}
    for path in args.result:
        result = json.loads(path.read_text())
        require(result['status']=='SCORER_RUN_FINISHED','Incomplete scorer cannot enter report')
        run = json.loads((path.parent/'run.json').read_text())
        row = dict(dataset=result['dataset'],split=result['split'],arm=result['arm'],samples=result['samples'],
            result=str(path.resolve()),result_sha256=sha(path),run_sha256=sha(path.parent/'run.json'),
            metrics=metric_values(result), native_identity=identity(result),
            sources_lock_sha256=run.get('sources_lock_sha256'),assets_lock_sha256=run.get('assets_lock_sha256'),
            own_code_sha256=run.get('own_code_sha256'),
            seed=run['arguments'].get('seed',112),replications=run['arguments'].get('replications',20),
            formal_official_result=False,quality_acceptance='PENDING')
        row['input_protocol']=result.get('input_protocol')
        row['feature_protocol']=result.get('feature_protocol')
        row['beat_profile']=result.get('beat_profile', 'public2025' if result['dataset']=='BEAT2' else None)
        row['evaluation_group']=('native_'+row['split']+'_vs_independent_reference_diagnostic'
            if row['arm']=='native_gt' and row['dataset'] in ('AIST++','FineDance') else row['arm'])
        row['metric_branch']='candidate_'+result['arm'] if 'metric_summary' in result else None
        row['confidence_interval_95']={name:values[row['metric_branch']]['confidence_interval_95']
                                      for name,values in result.get('metric_summary',{}).items()}
        rows.append(row)
        if row['arm']=='native_gt' and row['sources_lock_sha256'] and row['assets_lock_sha256'] and row['own_code_sha256']:
            key = (row['native_identity'],row['sources_lock_sha256'],row['assets_lock_sha256'],
                   json.dumps(row['own_code_sha256'],sort_keys=True),row['seed'],row['replications'])
            require(key not in originals,'Ambiguous native baseline for same protocol')
            originals[key]=row
    for row in rows:
        key=(row['native_identity'],row['sources_lock_sha256'],row['assets_lock_sha256'],
             json.dumps(row['own_code_sha256'],sort_keys=True),row['seed'],row['replications'])
        if row['arm']!='native_gt' and key in originals:
            base=originals[key]
            row['delta_from_native']={name: (np.asarray(numeric(value))-np.asarray(numeric(base['metrics'][name]))).tolist()
                                      for name,value in row['metrics'].items()}
            row['native_baseline_result']=base['result']
    write_json(args.output/'report.json',dict(status='SCORER_EVIDENCE_REPORTED_NOT_QUALITY_ACCEPTED',
                                             code_sha256=sha(__file__),rows=rows))
    lines=['# 官方打分器结果与转换对照','',
        '以下按原始Run列出。TRAIN为诊断；完整TEST运行也需另验协议和转换保真，不能凭分数生成通过结论。', '',
        '| 数据集 | split / arm | 源数 | 指标 | 证据 |','|---|---|---:|---|---|']
    def fmt(x):
        x=np.asarray(numeric(x))
        return ', '.join(f'{v:.6g}' for v in x.reshape(-1))
    for row in rows:
        metrics='；'.join(name+'='+fmt(value)+
            (' (95% CI ±'+fmt(row['confidence_interval_95'][name])+')' if name in row['confidence_interval_95'] else '')
            for name,value in row['metrics'].items())
        profile=(row.get('input_protocol') or {}).get('membership_profile')
        scope=row['split']+' / '+row['evaluation_group']+(' / '+profile if profile else '')
        if row['beat_profile']:
            scope+=' / '+row['beat_profile']
        if row['feature_protocol']:
            scope+=' / '+row['feature_protocol']['id']
        lines.append(f"| {row['dataset']} | {scope} | {row['samples']} | {metrics} | `{row['result']}` |")
        if 'delta_from_native' in row:
            delta='；'.join(name+'='+fmt(value) for name,value in row['delta_from_native'].items())
            lines.append(f"| {row['dataset']} | 配对差值（当前组−native GT） | {row['samples']} | {delta} | 同源、同参考池、同权重、同seed/repetitions |")
    lines += ['', '未匹配到同协议 native GT 的组不计算差值；不跨TRAIN/TEST、不同源集合或参考池作配对。',
              '舞蹈native GT行是当前split对参考池的诊断；FineDance论文GT主行对应LODGE发布202源池，见数据集评测说明.md。',
              'HumanML3D表格取candidate_<arm>分支；GT对照的native_gt分支独立采样caption/crop，数值可不同。95% CI来自作者的重复评测统计，不代表SOMA质量已经通过。',
              '作者实现若返回复数，JSON及表格均原样保留实部和虚部，不默默取实部或凭JSON层阈值决定通过；正式解释另依据数值检查。',
              'SOMA GT、Tokenizer和生成质量仍须各自的真实产物及冻结验收。']
    lines += ['BEAT2按caller及时间支持分组；public2025和emage2024_bc_l1不能混算差值。BC论文显示值为raw×10；旧版兼容不包含FGD。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='REPORT_WRITTEN',rows=len(rows),output=str(args.output))))


if __name__=='__main__': main()
