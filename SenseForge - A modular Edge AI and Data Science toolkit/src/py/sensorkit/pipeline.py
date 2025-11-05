import yaml, numpy as np, pandas as pd
from dataclasses import dataclass
from .io import read_csv, Columns
from . import features as F

@dataclass
class Config:
    sampling_hz: float
    win_size: int
    win_stride: int
    col_ts: str
    col_sig: str
    do_norm: bool = True
    @staticmethod
    def from_yaml(path:str):
        y=yaml.safe_load(open(path))
        d=y['data']
        return Config(
            sampling_hz=float(d.get('sampling_hz',200)),
            win_size=int(d['window']['size']),
            win_stride=int(d['window']['stride']),
            col_ts=d['columns']['timestamp'],
            col_sig=d['columns']['signal'],
            do_norm=bool(y.get('pipeline',{}).get('normalize',True)),
        )

class Pipeline:
    def __init__(self,cfg:Config): self.cfg=cfg
    @staticmethod
    def from_yaml(path:str): return Pipeline(Config.from_yaml(path))
    def _prep(self,df:pd.DataFrame):
        x=df[self.cfg.col_sig].to_numpy(dtype=np.float32)
        if self.cfg.do_norm:
            m=float(np.mean(x)); s=float(np.std(x)+1e-8); x=(x-m)/s
        return x
    def run(self,csv_path:str)->pd.DataFrame:
        df=read_csv(csv_path, Columns(self.cfg.col_ts,self.cfg.col_sig))
        x=self._prep(df)
        W=F.window_1d(x,self.cfg.win_size,self.cfg.win_stride)
        r=F.rms(W,axis=1); m, s = F.mean_std(W,axis=1); z=F.zcr(W,axis=1)
        return pd.DataFrame({'rms':r,'mean':m,'std':s,'zcr':z})
