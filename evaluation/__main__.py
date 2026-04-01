"""Usage: python -m evaluation SESSION_JSON [--judge --limit 3] [--output FILE]."""
import argparse
import hashlib
import json
from pathlib import Path
from .report import build_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--judge',action='store_true',help='Enable paid model judgments')
    parser.add_argument('--limit',type=int,default=5,help='Maximum participant turns sent to model judge')
    parser.add_argument('--model',default='gpt-4o-mini')
    args=parser.parse_args()
    if args.limit < 0: parser.error('--limit must be nonnegative')
    if args.output and args.output.resolve()==args.session.resolve(): parser.error('Output must differ from input session')
    try:
        raw=args.session.read_bytes()
        session=json.loads(raw)
        report=build_report(session,raw)
        report['evaluation_code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(Path(__file__).parent.glob('*.py'))}
        if args.judge:
            from dotenv import load_dotenv
            from langchain_openai import ChatOpenAI
            from .judge import evaluate_judges
            load_dotenv(Path(__file__).resolve().parents[1]/'.env')
            report['judge']=evaluate_judges(session,ChatOpenAI(model=args.model,temperature=0),args.model,args.limit)
        result=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            temporary=args.output.with_suffix(args.output.suffix+'.tmp')
            try:
                temporary.write_text(result,encoding='utf-8')
                temporary.replace(args.output)
            finally:
                temporary.unlink(missing_ok=True)
            print(f'Evaluation saved: {args.output}')
        else:
            print(result,end='')
    except (OSError,ValueError,TypeError,KeyError) as error:
        parser.error(str(error))


if __name__=='__main__':
    main()
