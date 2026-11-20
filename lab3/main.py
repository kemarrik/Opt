"""ЛР 3, вариант 9. Весь расчёт: python main.py."""
from pathlib import Path
import os,csv,json,sys,subprocess,time,platform
P=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(P/'results/.matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(P/'results/.cache'))
import numpy as np
import scipy
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from generate_data import make_instance
TOL=1e-7

def read_data():
    with (P/'data/nodes.csv').open(encoding='utf-8-sig') as f:nodes=list(csv.DictReader(f))
    n=len(nodes);cost=np.zeros((n,n))
    with (P/'data/arcs.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):cost[int(r['u']),int(r['v'])]=float(r['cost'])
    xy=np.array([[float(r['x_km']),float(r['y_km'])] for r in nodes])
    return xy,cost,json.loads((P/'data/config.json').read_text(encoding='utf-8'))

def write_csv(name,rows):
    if not rows:return
    with (P/'results'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save_json(name,r):
    (P/'results'/name).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def build_model(cost,closed=(),big_m=None,relax=False):
    n=len(cost);M=float(n-1 if big_m is None else big_m)
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j];m=len(arcs)
    row=[];col=[];val=[];lo=[];hi=[]
    def add(entries,l,u):
        r=len(lo)
        for j,v in entries:row.append(r);col.append(j);val.append(v)
        lo.append(l);hi.append(u)
    for node in range(n):
        add([(e,1) for e,(i,j) in enumerate(arcs) if i==node],1,1)
        add([(e,1) for e,(i,j) in enumerate(arcs) if j==node],1,1)
    for node in range(n):
        # Исходящий минус входящий искусственный поток.
        entries=[(m+e,(1 if i==node else -1)) for e,(i,j) in enumerate(arcs) if i==node or j==node]
        target=n-1 if node==0 else -1
        add(entries,target,target)
    for e in range(m):add([(m+e,1),(e,-M)],-np.inf,0)
    A=coo_matrix((val,(row,col)),shape=(len(lo),2*m)).tocsc()
    upper=np.r_[np.ones(m),np.full(m,n-1,dtype=float)]
    closed={tuple(v) for v in closed}
    for e,(i,j) in enumerate(arcs):
        if (i,j) in closed:upper[e]=0
        if j==0:upper[m+e]=0 # В корень искусственный поток не возвращается.
    objective=np.r_[[cost[i,j] for i,j in arcs],np.zeros(m)]
    integrality=np.r_[np.zeros(m) if relax else np.ones(m),np.zeros(m)].astype(int)
    return arcs,objective,integrality,Bounds(np.zeros(2*m),upper),LinearConstraint(A,np.array(lo),np.array(hi))

def verify(vector,cost,closed=(),big_m=None,integer=True):
    """Проверка по числовым переменным и исходным данным; без матриц ЛП и объекта солвера."""
    n=len(cost);M=n-1 if big_m is None else big_m
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j];m=len(arcs)
    x=np.asarray(vector[:m]);f=np.asarray(vector[m:]);closed={tuple(v) for v in closed}
    if not np.isfinite(vector).all():return {'passed':False,'reason':'nonfinite'}
    degree=[];flow=[]
    for node in range(n):
        outgoing=sum(x[e] for e,(i,j) in enumerate(arcs) if i==node)
        incoming=sum(x[e] for e,(i,j) in enumerate(arcs) if j==node)
        degree.extend([abs(outgoing-1),abs(incoming-1)])
        balance=sum(f[e] for e,(i,j) in enumerate(arcs) if i==node)-sum(f[e] for e,(i,j) in enumerate(arcs) if j==node)
        flow.append(abs(balance-(n-1 if node==0 else -1)))
    errors={'degree_error':float(max(degree)),'flow_balance_error':float(max(flow)),
            'link_violation':float(max(0,np.max(f-M*x))),
            'bound_violation':float(max(0,-x.min(),x.max()-1,-f.min(),f.max()-(n-1))),
            'closed_arc_violation':float(max([0]+[float(x[e]) for e,a in enumerate(arcs) if a in closed])),
            'root_flow_violation':float(max([0]+[abs(float(f[e])) for e,(i,j) in enumerate(arcs) if j==0]))}
    errors['integer_error']=float(np.max(np.abs(x-np.rint(x)))) if integer else 0
    route=[]
    if integer and errors['integer_error']<=TOL:
        successors={i:j for e,(i,j) in enumerate(arcs) if x[e]>.5}
        route=[0]
        for _ in range(n):
            if route[-1] not in successors:break
            route.append(successors[route[-1]])
        single=len(route)==n+1 and route[-1]==0 and len(set(route[:-1]))==n
    else:single=None
    objective=sum(float(x[e])*cost[i,j] for e,(i,j) in enumerate(arcs))
    return {'passed':bool(max(errors.values())<=TOL and (not integer or single)),**errors,
            'objective_recomputed':float(objective),'single_tour':single,'route':route}

def vector_from_route(route,n):
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j];index={a:e for e,a in enumerate(arcs)}
    v=np.zeros(2*len(arcs))
    for k,(i,j) in enumerate(zip(route,route[1:])):
        e=index[(i,j)];v[e]=1;v[len(arcs)+e]=n-1-k
    return v

def greedy(cost,closed=()):
    """Ближайший непосещённый сосед, повтор из каждой стартовой вершины; без возврата и обменов."""
    n=len(cost);closed={tuple(v) for v in closed};best=None
    for start in range(n):
        route=[start];remaining=set(range(n))-{start}
        while remaining:
            choices=[j for j in remaining if (route[-1],j) not in closed]
            if not choices:break
            j=min(choices,key=lambda j:(cost[route[-1],j],j));route.append(j);remaining.remove(j)
        if remaining or (route[-1],start) in closed:continue
        route.append(start)
        price=float(sum(cost[i,j] for i,j in zip(route,route[1:])))
        if best is None or price<best[0]:
            cycle=route[:-1];pos=cycle.index(0);cycle=cycle[pos:]+cycle[:pos];best=(price,cycle+[0])
    return best

def optimize(cost,closed=(),big_m=None,relax=False,time_limit=30):
    arcs,c,integ,bounds,cons=build_model(cost,closed,big_m,relax)
    start=time.perf_counter()
    result=milp(c,integrality=integ,bounds=bounds,constraints=cons,
                options={'disp':True,'presolve':True,'time_limit':time_limit,'mip_rel_gap':0.0})
    elapsed=time.perf_counter()-start
    out={'status':int(result.status),'message':result.message,'seconds':elapsed,'n':len(cost),
         'M':float(len(cost)-1 if big_m is None else big_m),'relaxation':relax}
    for attr in ['fun','mip_gap','mip_dual_bound','mip_node_count']:
        value=getattr(result,attr,None)
        out[attr]=float(value) if value is not None and np.isfinite(value) else None
    if result.x is not None:
        check=verify(result.x,cost,closed,big_m,not relax)
        check['objective_difference']=abs(check['objective_recomputed']-float(result.fun))
        out['verification']=check
        assert check['passed'] and check['objective_difference']<=TOL
        rows=[{'u':i,'v':j,'x':float(result.x[e]),'f':float(result.x[len(arcs)+e]),'cost':float(cost[i,j]),'allowed':(i,j) not in {tuple(z) for z in closed}} for e,(i,j) in enumerate(arcs)]
    else:rows=[]
    return out,rows

def plot_routes(xy,rows,route,name,price):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6))
    for r in rows:
        if r['x']<=1e-6:continue
        i,j=r['u'],r['v'];a,b=xy[i],xy[j]
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#276388','lw':1.4 if route else .6+2*r['x'],'shrinkA':3,'shrinkB':3,'linestyle':'--' if r['x']<.999 else '-','connectionstyle':'arc3,rad=0.06'})
    ax.scatter(xy[:,0],xy[:,1],s=45,c='#244357',zorder=3)
    offsets={0:(-12,-13),6:(12,10),11:(-10,10),9:(10,-13),10:(-10,9),3:(10,-9)}
    for i,(x,y) in enumerate(xy):
        ax.annotate(str(i),(x,y),xytext=offsets.get(i,(7,8)),textcoords='offset points',ha='center',va='center',fontsize=10,zorder=4,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    titles={'base':'Базовый маршрут','deficit':'Закрыто 26 дорог','stress':'Закрыто 52 дороги','base_lp':'LP-релаксация: дробные дуги'}
    ax.set_title(f'{titles[name]}: {price:.2f} усл. ед.');ax.set_xlabel('x, км');ax.set_ylabel('y, км');ax.margins(.12);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(P/f'results/{name}_route.png',dpi=160);plt.close(fig)

def worker(mode,name):
    xy,cost,cfg=read_data()
    if mode=='scenario':
        closed=cfg['scenarios'][name]['closed'];out={}
        for relax in [True,False]:
            key='lp' if relax else 'milp';r,rows=optimize(cost,closed,relax=relax,time_limit=cfg['solver']['time_limit'])
            out[key]=r;write_csv(f'{name}_{key}_solution.csv',rows)
            if rows and (not relax or name=='base'):plot_routes(xy,rows,r['verification']['route'],name+('_lp' if relax else ''),r['fun'])
        start=time.perf_counter();g=greedy(cost,closed);seconds=time.perf_counter()-start
        out['greedy']={'seconds':seconds,'complete':g is not None}
        if g:
            vec=vector_from_route(g[1],len(cost));check=verify(vec,cost,closed);assert check['passed']
            out['greedy'].update({'objective':g[0],'route':g[1],'verification':check})
            write_csv(name+'_greedy_solution.csv',[{'u':i,'v':j,'cost':float(cost[i,j])} for i,j in zip(g[1],g[1][1:])])
        if out['lp']['fun'] is not None and out['milp']['fun'] is not None:
            out['integrality_gap_percent']=100*(out['milp']['fun']-out['lp']['fun'])/out['milp']['fun']
        if g and out['milp']['fun'] is not None:out['greedy']['gap_percent']=100*(g[0]-out['milp']['fun'])/out['milp']['fun']
        save_json(f'{name}_summary.json',out)
    elif mode=='bigm':
        M=int(name);out={}
        for relax in [True,False]:
            key='lp' if relax else 'milp';r,rows=optimize(cost,big_m=M,relax=relax,time_limit=30)
            out[key]=r;write_csv(f'M{M}_{key}_solution.csv',rows)
        save_json(f'M{M}_summary.json',out)
    elif mode=='benchmark':
        n=int(name);xy,cost=make_instance(n,10006);out={}
        write_csv(f'benchmark_{n}_nodes.csv',[{'node':i,'x_km':float(p[0]),'y_km':float(p[1])} for i,p in enumerate(xy)])
        write_csv(f'benchmark_{n}_costs.csv',[{'u':i,'v':j,'cost':float(cost[i,j])} for i in range(n) for j in range(n) if i!=j])
        for relax in [True,False]:
            key='lp' if relax else 'milp';r,rows=optimize(cost,relax=relax,time_limit=cfg['benchmark']['time_limit'])
            out[key]=r;write_csv(f'benchmark_{n}_{key}_solution.csv',rows)
        start=time.perf_counter();g=greedy(cost);elapsed=time.perf_counter()-start
        out['greedy']={'objective':g[0],'route':g[1],'seconds':elapsed}
        save_json(f'benchmark_{n}_summary.json',out)

def run_worker(mode,name):
    with (P/f'results/{mode}_{name}_solver.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',mode,str(name)],stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    (P/'results').mkdir(exist_ok=True)
    _,_,cfg=read_data()
    for name in cfg['scenarios']:
        run_worker('scenario',name);print(f'{name}: готово',flush=True)
    run_worker('bigm',11000)
    for n in cfg['benchmark']['sizes']:
        run_worker('benchmark',n);print(f'{n} вершин: готово',flush=True)
    summarize(cfg)

def summarize(cfg):
    scenarios=[]
    for name in cfg['scenarios']:
        r=json.loads((P/f'results/{name}_summary.json').read_text())
        scenarios.append({'scenario':name,'lp':r['lp']['fun'],'milp':r['milp']['fun'],'status':r['milp']['status'],
                          'integrality_gap_percent':r.get('integrality_gap_percent'),'solver_gap':r['milp']['mip_gap'],
                          'greedy':r['greedy'].get('objective'),'greedy_gap_percent':r['greedy'].get('gap_percent')})
    write_csv('scenarios.csv',scenarios)
    bench=[]
    for n in cfg['benchmark']['sizes']:
        r=json.loads((P/f'results/benchmark_{n}_summary.json').read_text());m=r['milp']
        bench.append({'n':n,'status':m['status'],'seconds':m['seconds'],'lp_bound':r['lp']['fun'],'incumbent':m['fun'],
                      'solver_dual_bound':m['mip_dual_bound'],'solver_gap_percent':None if m['mip_gap'] is None else m['mip_gap']*100,
                      'nodes':m['mip_node_count'],'greedy':r['greedy']['objective'],
                      'lp_gap_percent':None if m['fun'] is None else 100*(m['fun']-r['lp']['fun'])/m['fun'],
                      'greedy_seconds':r['greedy']['seconds']})
    write_csv('benchmark.csv',bench)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].plot([r['n'] for r in bench],[r['seconds'] for r in bench],'o-');axes[0].set_ylabel('Время, с')
    axes[1].plot([r['n'] for r in bench],[np.nan if r['lp_gap_percent'] is None else r['lp_gap_percent'] for r in bench],'o-');axes[1].set_ylabel('Разница MILP и LP, %')
    axes[2].plot([r['n'] for r in bench],[np.nan if r['solver_gap_percent'] is None else r['solver_gap_percent'] for r in bench],'o-');axes[2].set_ylabel('Зазор ветвления, %')
    for ax in axes:ax.set_xlabel('Число вершин');ax.set_xticks(cfg['benchmark']['sizes']);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(P/'results/benchmark.png',dpi=160);plt.close(fig)
    save_json('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'solver':'HiGHS; версия в логах'})
    print(json.dumps(scenarios,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3])
    else:main()
