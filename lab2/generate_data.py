"""Воспроизводимая сеть из 10 узлов и 20 дуг. python generate_data.py"""
from pathlib import Path
import csv,json
import numpy as np
P=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(10006)
    # u,v,capacity,base cost: общие мощности, отдельные тарифы грузов.
    topology=[(0,2,14,2),(0,3,12,3),(1,2,14,2),(1,3,12,3),
              (2,4,14,1),(2,5,14,5),(3,4,12,3),(3,5,12,2),
              (4,6,14,1),(4,7,14,4),(5,6,14,3),(5,7,14,2),
              (6,8,14,2),(6,9,14,2),(7,8,14,3),(7,9,14,3),
              (2,3,8,2),(4,5,8,2),(6,7,8,2),(0,5,8,8)]
    with (P/'data/arcs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['arc','u','v','capacity','cost_A','cost_B'])
        for i,(u,v,cap,c) in enumerate(topology):
            w.writerow([f'e{i+1:02}',u,v,cap,round(c*rng.uniform(.8,1.2),2),round(c*rng.uniform(.8,1.2),2)])
    config={'variant':9,'N':6,'Q':0,'Y':26,'seed':10006,'node_count':10,
            'commodities':[{'name':'A','source':0,'sink':8,'demand':10},
                           {'name':'B','source':1,'sink':9,'demand':12}]}
    (P/'data/config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
