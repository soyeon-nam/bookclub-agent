"""Run synthetic contrast and repeated-judge checks; opt in to paid calls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor
from .semantic import evaluate_semantic

FIXTURES = Path(__file__).parent / 'fixtures' / 'validity'


def fixture_session(case, variant):
    return {'schema_version':1, 'session_id':'synthetic', 'book':case['book'],
            'participants':case['participants'], 'user_name':'Reader', 'metadata':{}, 'discarded':[],
            'utterances':[dict(turn, turn_id='turn-%02d'%index, kind='participant', evidence={})
                          for index, turn in enumerate(case['variants'][variant])]}


def repeated_summary(records):
    values = [r['mean'] for r in records if r['status']=='ok' and r['mean'] is not None]
    return {'requested_runs':len(records), 'valid_runs':len(values), 'values':values,
            'mean':statistics.mean(values) if values else None,
            'range':max(values)-min(values) if len(values)>1 else None,
            'stdev':statistics.stdev(values) if len(values)>1 else None,
            'complete':len(values)==len(records) and len(values)>1}


def run_case(case, llm, model, repeats=2):
    if type(repeats) is not int or not 2 <= repeats <= 10:
        raise ValueError('Use 2–10 repeated judgments')
    records = {variant:[] for variant in case['variants']}
    for iteration in range(repeats):
        order = list(records) if iteration % 2 == 0 else list(reversed(records))
        for variant in order:
            records[variant].append(evaluate_semantic(fixture_session(case,variant),case['metric'],llm,model))
    summaries = {key:repeated_summary(value) for key,value in records.items()}
    contrasts = {}
    for variant in records:
        if variant == 'good':
            continue
        good, bad = summaries['good'], summaries[variant]
        complete = good['complete'] and bad['complete']
        gap = good['mean']-bad['mean'] if complete else None
        contrasts[variant] = {'gap':gap, 'minimum_gap':case['minimum_gap'],
                             'verdict':('pass' if gap >= case['minimum_gap'] else 'fail') if complete else 'incomplete'}
    return {'id':case['id'], 'metric':case['metric'], 'records':records,
            'repeatability':summaries, 'contrasts':contrasts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--model',default='gpt-4o-mini')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 2 <= args.repeats <= 10: parser.error('Use 2–10 repeats')
    paths=sorted(FIXTURES.glob('*.json'))
    if args.only:
        if set(args.only)-{p.stem for p in paths}: parser.error('Unknown case IDs')
        paths=[p for p in paths if p.stem in args.only]
    if args.output.resolve() in {p.resolve() for p in paths}: parser.error('Output must differ from fixtures')
    cases=[json.loads(p.read_text()) for p in paths]
    result={'status':'completed' if args.live else 'plan_only','model':args.model,'temperature':0,'repeats':args.repeats,
            'planned_judgments':sum(len(c['variants'])*args.repeats for c in cases),
            'maximum_model_invocations':2*sum(len(c['variants'])*args.repeats for c in cases),
            'fixture_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'limitations':['Synthetic contrast checks are not human criterion validity.',
                          'Repeatability is not accuracy. Small samples and same-family model bias remain.',
                          'Synthetic book assertions are supplied text, not verified facts about the original book.',
                          'Failed or unavailable runs prevent a passing contrast; no score imputation.']}
    if args.live:
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
        load_dotenv(Path(__file__).resolve().parents[1]/'.env')
        llm=ChatOpenAI(model=args.model,temperature=0,request_timeout=60,max_retries=1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result['cases']=list(pool.map(lambda c:run_case(c,llm,args.model,args.repeats),cases))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Saved:',args.output)


if __name__=='__main__': main()
