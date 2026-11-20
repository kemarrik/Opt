from pathlib import Path
import sys,csv,json,hashlib,subprocess
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from main import read_data,verify,vector_from_route
xy,cost,cfg=read_data();n=len(cost);arcs=[(i,j) for i in range(n) for j in range(n) if i!=j]
def read_vector(name):
 with (P/f'results/{name}').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 return np.array([float(r['x']) for r in rows]+[float(r['f']) for r in rows])
def held_karp(c,closed):
 # Независимый точный алгоритм, только для проверки малых основных задач.
 closed={tuple(a) for a in closed};N=len(c);dp={}
 for j in range(1,N):
  if (0,j) not in closed:dp[(1<<(j-1),j)]=float(c[0,j])
 for mask in range(1,1<<(N-1)):
  for j in range(1,N):
   if (mask,j) not in dp:continue
   value=dp[(mask,j)]
   for k in range(1,N):
    if mask&(1<<(k-1)) or (j,k) in closed:continue
    key=(mask|(1<<(k-1)),k);new=value+c[j,k]
    if new<dp.get(key,float('inf')):dp[key]=new
 full=(1<<(N-1))-1
 return min(dp[(full,j)]+c[j,0] for j in range(1,N) if (full,j) in dp and (j,0) not in closed)
checks={}
for name in cfg['scenarios']:
 closed=cfg['scenarios'][name]['closed'];r=json.loads((P/f'results/{name}_summary.json').read_text())
 v=read_vector(name+'_milp_solution.csv');assert verify(v,cost,closed)['passed']
 assert verify(read_vector(name+'_lp_solution.csv'),cost,closed,integer=False)['passed']
 exact=held_karp(cost,closed)
 assert abs(exact-r['milp']['fun'])<1e-7
 assert r['milp']['status']==0 and r['milp']['mip_gap']==0
 checks[name]={'valid':True,'held_karp_optimum':exact,'milp_difference':abs(exact-r['milp']['fun'])}
# Проверка детектора раздельных циклов.
v=np.zeros(2*len(arcs));cycles=[[0,1,2,3,4,5,0],[6,7,8,9,10,11,6]]
index={a:e for e,a in enumerate(arcs)}
for route in cycles:
 for a in zip(route,route[1:]):v[index[a]]=1
bad=verify(v,cost);assert not bad['passed'] and not bad['single_tour'] and bad['degree_error']==0
checks['subtours_rejected']=True
v=read_vector('base_milp_solution.csv');v[len(arcs):]=0
assert not verify(v,cost)['passed'];checks['bad_artificial_flow_rejected']=True
v=read_vector('base_lp_solution.csv');assert not verify(v,cost,integer=True)['passed'];checks['fractional_plan_rejected_as_tour']=True
v=read_vector('base_milp_solution.csv');v[0]=-.1
assert not verify(v,cost)['passed'];checks['negative_value_rejected']=True
# Неподходящая дорога должна быть запрещена даже при правильном маршруте.
v=read_vector('base_milp_solution.csv');e=next(e for e in range(len(arcs)) if v[e]>.5)
assert not verify(v,cost,[arcs[e]])['passed'];checks['closed_arc_rejected']=True
before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'data').iterdir()}
subprocess.run([sys.executable,str(P/'generate_data.py')],check=True)
assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'data').iterdir()}
checks['seed_reproducible']=True
for size in [10,20,40]:
 r=json.loads((P/f'results/benchmark_{size}_summary.json').read_text())
 assert r['milp']['verification']['passed'];assert r['lp']['verification']['passed']
 checks[f'benchmark_{size}_valid']=True
(P/'results/validation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
