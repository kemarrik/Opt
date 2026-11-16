"""Лабораторная 2, вариант 9. Полный расчёт: python main.py."""
from pathlib import Path
import csv,json,sys,subprocess,platform,heapq,time,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'results/.matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'results/.cache'))
import numpy as np
import scipy
from scipy.optimize import linprog

P=Path(__file__).resolve().parent
TOL=1e-7

def read_data():
    with (P/'data/arcs.csv').open(encoding='utf-8-sig') as f:arcs=list(csv.DictReader(f))
    for a in arcs:
        for k in ['u','v']:a[k]=int(a[k])
        for k in ['capacity','cost_A','cost_B']:a[k]=float(a[k])
    return arcs,json.loads((P/'data/config.json').read_text(encoding='utf-8'))

def write_csv(name,rows):
    with (P/'results'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build_model(arcs,cfg,factor=1):
    m,n=len(arcs),cfg['node_count'];K=cfg['commodities']
    c=np.array([a['cost_'+k['name']] for k in K for a in arcs])
    eq=np.zeros((len(K)*n,len(K)*m));b=np.zeros(len(K)*n)
    for j,k in enumerate(K):
        b[j*n+k['source']]=factor*k['demand'];b[j*n+k['sink']]=-factor*k['demand']
        for e,a in enumerate(arcs):eq[j*n+a['u'],j*m+e]=1;eq[j*n+a['v'],j*m+e]=-1
    cap=np.tile(np.eye(m),(1,len(K)))
    return c,cap,np.array([a['capacity'] for a in arcs]),eq,b

def verify(x,arcs,cfg,factor=1):
    """Проверка напрямую по таблице дуг, без матриц модели и объекта солвера."""
    balances=[];capacities=[];objective=0
    for j,k in enumerate(cfg['commodities']):
        for node in range(cfg['node_count']):
            actual=sum(float(x[j,e]) for e,a in enumerate(arcs) if a['u']==node)-sum(float(x[j,e]) for e,a in enumerate(arcs) if a['v']==node)
            expected=factor*k['demand']*((node==k['source'])-(node==k['sink']))
            balances.append({'commodity':k['name'],'node':node,'out_minus_in':actual,'required':expected,'residual':actual-expected})
        objective+=sum(float(x[j,e])*a['cost_'+k['name']] for e,a in enumerate(arcs))
    for e,a in enumerate(arcs):
        total=float(x[:,e].sum());capacities.append({'arc':a['arc'],'u':a['u'],'v':a['v'],'flow_A':float(x[0,e]),'flow_B':float(x[1,e]),'total':total,'capacity':a['capacity'],'slack':a['capacity']-total})
    bal=max(abs(r['residual']) for r in balances);viol=max(0,max(-r['slack'] for r in capacities));neg=max(0,-float(x.min()))
    return {'passed':bool(max(bal,viol,neg)<=TOL),'balance_error':bal,'capacity_violation':viol,'nonnegativity_violation':neg,'objective':objective},balances,capacities

def greedy(arcs,cfg,factor=1):
    """Последовательно A затем B, кратчайший по тарифу путь в остаточной сети; без отмены ранее выбранных потоков."""
    residual=np.array([a['capacity'] for a in arcs]);x=np.zeros((2,len(arcs)))
    for j,k in enumerate(cfg['commodities']):
        remaining=factor*k['demand']
        while remaining>TOL:
            dist={k['source']:0};prev={};q=[(0,k['source'])]
            while q:
                cost,u=heapq.heappop(q)
                if cost>dist[u]+1e-12:continue
                for e,a in enumerate(arcs):
                    if a['u']!=u or residual[e]<=TOL:continue
                    v=a['v'];new=cost+a['cost_'+k['name']]
                    if new<dist.get(v,float('inf'))-1e-12:
                        dist[v]=new;prev[v]=(u,e);heapq.heappush(q,(new,v))
            if k['sink'] not in dist:return x,False
            path=[];v=k['sink']
            while v!=k['source']:u,e=prev[v];path.append(e);v=u
            amount=min(remaining,min(residual[e] for e in path))
            for e in path:x[j,e]+=amount;residual[e]-=amount
            remaining-=amount
    return x,True

def plot_network(arcs,flows,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    pos={0:(0,1),1:(0,-1),2:(1.5,1),3:(1.5,-1),4:(3,1),5:(3,-1),6:(4.5,1),7:(4.5,-1),8:(6,1),9:(6,-1)}
    fig,ax=plt.subplots(figsize=(10,5));ax.set_xlim(-.6,6.6);ax.set_ylim(-2,2);ax.axis('off')
    for e,a in enumerate(arcs):
        u,v=pos[a['u']],pos[a['v']];f=np.maximum(flows[:,e],0);active=f.sum()>TOL
        rad=.38 if (a['u'],a['v'])==(0,5) else 0
        patch=FancyArrowPatch(u,v,arrowstyle='-|>',mutation_scale=13,shrinkA=18,shrinkB=18,
                              connectionstyle=f'arc3,rad={rad}',linewidth=1+f.sum()/7,color='#12689a' if active else '#a8a8a8',zorder=1)
        ax.add_patch(patch)
        x,y=(u[0]+v[0])/2,(u[1]+v[1])/2
        if u[1]!=v[1] and u[0]!=v[0]:x+=.17 if u[1]>v[1] else -.17;y+=.25 if u[1]>v[1] else -.25
        elif u[1]==v[1]:y+=.16 if u[1]>0 else -.19
        else:x+=.30
        if rad:x,y=1.0,-1.65
        ax.text(x,y,f"{a['arc']}: {f[0]:.1f}/{f[1]:.1f} | {a['capacity']:g}",fontsize=8.5,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1.5},zorder=4)
    for node,(x,y) in pos.items():
        ax.scatter([x],[y],s=700,color='#e2edf5',edgecolor='#244357',zorder=5);ax.text(x,y,str(node),ha='center',va='center',zorder=6,fontsize=12)
    titles={'base':'Базовый план','growth':'Рост спроса на 20%','overload':'Снижение мощности e05 на 20%','failure':'Отказ дуги e05'}
    ax.set_title(titles[name]+'\nПодписи: дуга, поток A / поток B | общая мощность (т/сут.)',fontsize=10)
    fig.tight_layout();fig.savefig(P/f'results/{name}_network.png',dpi=180);plt.close(fig)

def worker(name):
    arcs,cfg=read_data();factor=1
    base=json.loads((P/'results/base_summary.json').read_text()) if name!='base' else None
    if name=='growth':factor=1.2
    if name in ['failure','overload']:
        target=base['most_loaded_arc']
        for a in arcs:
            if a['arc']==target:a['capacity']*=0 if name=='failure' else .8
    c,A,b,E,d=build_model(arcs,cfg,factor)
    start=time.perf_counter();res=linprog(c,A_ub=A,b_ub=b,A_eq=E,b_eq=d,bounds=(0,None),method='highs-ds',options={'disp':True,'presolve':True,'primal_feasibility_tolerance':TOL,'dual_feasibility_tolerance':TOL});elapsed=time.perf_counter()-start
    out={'scenario':name,'status':int(res.status),'message':res.message,'solver_seconds':elapsed,'iterations':int(res.nit)}
    if res.success:
        x=res.x.reshape(2,-1);check,bal,cap=verify(x,arcs,cfg,factor)
        check['objective_difference']=abs(check['objective']-float(res.fun));assert check['passed'] and check['objective_difference']<=TOL
        out.update(check);out['most_loaded_arc']=max(cap,key=lambda r:(r['total'],-int(r['arc'][1:])))['arc']
        for i,r in enumerate(cap):r.update({'utilization':r['total']/r['capacity'] if r['capacity'] else 0,'saturated':r['capacity']>0 and abs(r['slack'])<=TOL,'shadow_price':float(res.ineqlin.marginals[i])})
        write_csv(name+'_flows.csv',cap);write_csv(name+'_balances.csv',bal)
        # Проверка локальных цен всех насыщенных дуг.
        sensitivity=[]
        for i,r in enumerate(cap):
            if not r['saturated']:continue
            b2=b.copy();b2[i]+=.001
            rr=linprog(c,A_ub=A,b_ub=b2,A_eq=E,b_eq=d,bounds=(0,None),method='highs-ds')
            if rr.success:sensitivity.append({'arc':r['arc'],'capacity_delta':.001,'prediction':r['shadow_price']*.001,'actual':float(rr.fun-res.fun)})
            if b[i]>=.001:
                b3=b.copy();b3[i]-=.001
                rr=linprog(c,A_ub=A,b_ub=b3,A_eq=E,b_eq=d,bounds=(0,None),method='highs-ds')
                if rr.success:sensitivity.append({'arc':r['arc'],'capacity_delta':-.001,'prediction':r['shadow_price']*(-.001),'actual':float(rr.fun-res.fun)})
        if sensitivity:write_csv(name+'_sensitivity.csv',sensitivity)
        start=time.perf_counter();gx,ok=greedy(arcs,cfg,factor);gt=time.perf_counter()-start
        gc,gb,gcap=verify(gx,arcs,cfg,factor)
        out['greedy']={'complete':ok,'seconds':gt,**gc}
        if ok and gc['passed']:
            out['greedy']['gap_percent']=100*(gc['objective']-res.fun)/res.fun
            write_csv(name+'_greedy_flows.csv',gcap)
        plot_network(arcs,x,name)
    (P/f'results/{name}_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    (P/'results').mkdir(exist_ok=True)
    results=[]
    for name in ['base','overload','growth','failure']:
        with (P/f'results/{name}_solver.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name],stdout=f,stderr=subprocess.STDOUT,check=True)
        results.append(json.loads((P/f'results/{name}_summary.json').read_text()))
    write_csv('scenarios.csv',[{'scenario':r['scenario'],'status':r['status'],'objective':r.get('objective',''),'lp_seconds':r['solver_seconds'],'greedy_objective':r.get('greedy',{}).get('objective',''),'greedy_complete':r.get('greedy',{}).get('complete',False),'greedy_gap_percent':r.get('greedy',{}).get('gap_percent','')} for r in results])
    (P/'results/environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'solver':'HiGHS; version in logs'},indent=2))
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
