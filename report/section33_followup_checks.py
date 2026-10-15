import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from section33.datasets import load_split
from section33.analysis.power import paired_development_distribution
from section33.analysis.evaluation import paired_within_question_bootstrap
from section33.data.cohort import validate_cohort, RELATIONS
from section33.cli import cmd_setup, _verify_protocol
from section33.config import run_paths

out = {}
with tempfile.TemporaryDirectory() as tmp:
    paths={k:Path(tmp)/k for k in ('labels','features')}
    for p in paths.values(): p.mkdir()
    base=dict(question_id='q',subject_id='s',split='test',track='A',answer_index=0,candidate_key='k')
    labels=[dict(base,label=1,multiplicity=1)]
    features=[dict(base,split='train',position='last_prompt',candidate_token_ids=[1],content_length=1,multiplicity=1,f__dummy=1)]
    confidence=[dict(base,value=80,parse_ok=True,raw_text='80')]*2
    for path,rows in [(paths['labels']/'trackA_answers__stub.jsonl',labels),(paths['labels']/'trackA_confidence__stub.jsonl',confidence),(paths['features']/'trackA_features__stub.jsonl',features)]:
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=load_split(paths,'stub','A')
    out['join_integrity']={'input_label_rows':1,'output_rows':len(result),'label_split':result['split'].tolist(),'feature_split':result['split_feat'].tolist()}

rows=[]
for i,delta in enumerate([1,0,-1]):
    for y in (0,1):
        rows.append(dict(subject_id=str(i),question_id=str(i),label=y,a=float(y) if delta>=0 else float(1-y),b=float(y) if delta<=0 else float(1-y),relation='father',stratum='0'))
df=pd.DataFrame(rows)
dist=paired_development_distribution(df,'a','b',n_boot=300,seed=4)
rng=np.random.default_rng(4)
expected=np.array([np.mean([{'0':1.,'1':0.,'2':-1.}[s] for s in rng.choice(['0','1','2'],size=3,replace=True)]) for _ in range(300)])
out['within_power_bootstrap']={'draws':300,'mismatched_draws':int(np.sum(~np.isclose(expected,dist['delta_within']))),'expected_sd':float(expected.std(ddof=1)),'actual_sd':float(dist['delta_within'].std(ddof=1))}
try:
    paired_within_question_bootstrap(df.assign(label=1),'a','b',n_boot=5)
except Exception as e:
    out['zero_eligible_questions']={'error':type(e).__name__,'message':str(e)}

small=pd.DataFrame([{'relation':r,'split':split,'subject_id':f'{r}/{split}'} for r in RELATIONS for split in ('train','dev','calibration','test')])
out['confirmatory_tiny_cohort']=validate_cohort(small,{s:4 for s in ('train','dev','calibration','test')},mode='confirmatory',excluded_subjects=1)
with tempfile.TemporaryDirectory() as tmp:
    cmd_setup(SimpleNamespace(run=Path(tmp)))
    paths=run_paths(Path(tmp))
    _verify_protocol(paths)
    out['protocol_gate']={'passes':True,'cohort_manifest_present':(paths['manifests']/'cohort_manifest.json').exists(),'power_files':len(list(paths['protocol'].glob('power*.json')))}
print(json.dumps(out,indent=2))
