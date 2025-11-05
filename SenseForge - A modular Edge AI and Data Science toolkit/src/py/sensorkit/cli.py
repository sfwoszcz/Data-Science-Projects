import argparse, os, pandas as pd, matplotlib.pyplot as plt
from .pipeline import Pipeline

def cmd_summarize(a):
    p=Pipeline.from_yaml(a.config); df=p.run(a.csv); print(df.describe())

def cmd_plot(a):
    p=Pipeline.from_yaml(a.config); df=pd.read_csv(a.csv);
    os.makedirs('artifacts',exist_ok=True)
    fig=plt.figure(); ax=fig.add_subplot(111)
    ax.plot(df.iloc[:2000][p.cfg.col_sig].values); ax.set_title('Signal preview')
    fig.savefig(a.out,dpi=160); print(f'Saved {a.out}')

def cmd_features(a):
    p=Pipeline.from_yaml(a.config); df=p.run(a.csv)
    out=a.out or 'artifacts/features.csv'
    os.makedirs(os.path.dirname(out),exist_ok=True); df.to_csv(out,index=False)
    print(f'Wrote {out}')

def build_parser():
    p=argparse.ArgumentParser(prog='sensorkit',description='Sensor Log Toolkit')
    p.add_argument('--config',default='configs/default.yaml')
    s=p.add_subparsers(dest='cmd',required=True)
    s1=s.add_parser('summarize'); s1.add_argument('csv'); s1.set_defaults(func=cmd_summarize)
    s2=s.add_parser('plot'); s2.add_argument('csv'); s2.add_argument('--out',default='artifacts/eda.png'); s2.set_defaults(func=cmd_plot)
    s3=s.add_parser('features'); s3.add_argument('csv'); s3.add_argument('--out',default='artifacts/features.csv'); s3.set_defaults(func=cmd_features)
    return p

def main():
    a=build_parser().parse_args(); a.func(a)
