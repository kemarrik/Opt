"""Учебные данные TSP. Повторная генерация: python generate_data.py."""
from pathlib import Path
import csv,json
import numpy as np
P=Path(__file__).resolve().parent

def make_instance(n, seed):
    rng=np.random.default_rng(seed)
    xy=np.round(rng.uniform(0,100,(n,2)),2)
    costs=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:costs[i,j]=round(float(np.linalg.norm(xy[i]-xy[j]))*rng.uniform(1,1.25),2)
    return xy,costs

def main():
    n=12;xy,c=make_instance(n,10006)
    with (P/'data/nodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['node','x_km','y_km']);w.writerows((i,*xy[i]) for i in range(n))
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j]
    with (P/'data/arcs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['u','v','cost']);w.writerows((i,j,c[i,j]) for i,j in arcs)
    rng=np.random.default_rng(10007);order=rng.permutation(len(arcs))
    closed=[[int(arcs[k][0]),int(arcs[k][1])] for k in order[:52]]
    config={'variant':9,'group':'Б24-507','author':'Демидович В.А.','N':6,'Q':0,'Y':26,'seed':10006,'node_count':n,
            'scenarios':{'base':{'closed':[]},'deficit':{'closed':closed[:26]},'stress':{'closed':closed}},
            'solver':{'time_limit':30,'mip_rel_gap':0.0},'benchmark':{'sizes':[10,20,40],'time_limit':8}}
    (P/'data/config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
