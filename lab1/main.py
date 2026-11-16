"""ЛР 1, вариант 9. Запуск: python main.py. Данные не изменяются при решении."""
from pathlib import Path
import csv, json, sys, subprocess, platform
import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
TOL = 1e-7

def load_data():
    with (ROOT/'data/dishes.csv').open(encoding='utf-8-sig') as f:
        dishes = list(csv.DictReader(f))
    return dishes, json.loads((ROOT/'data/config.json').read_text(encoding='utf-8'))

def build_model(dishes, config, scenario):
    n, days = len(dishes), config['days']
    c = np.tile([float(d['cost_rub']) for d in dishes], days)
    rows, rhs, labels = [], [], []
    for day in range(days):
        for nutrient, limits in config['daily_limits'].items():
            row = np.zeros(n*days)
            row[day*n:(day+1)*n] = [float(d[nutrient]) for d in dishes]
            for sign, bound, kind in [(1, limits[1], 'max'), (-1, limits[0], 'min')]:
                rows.append(sign*row); rhs.append(sign*bound)
                labels.append(f'day{day+1}_{nutrient}_{kind}')
    rows.append(np.ones(n*days)); rhs.append(scenario['portion_capacity']); labels.append('weekly_portions_max')
    return c, np.array(rows), np.array(rhs, dtype=float), labels

def verify(x, c, A, b, upper_bound=3):
    # Проверка использует только исходные матрицы и числовой вектор, не объект OptimizeResult.
    slack = b-A@x
    lower = float(np.min(x))
    return {'passed': bool(np.all(slack >= -TOL) and lower >= -TOL and np.all(x <= upper_bound+TOL)),
            'max_constraint_violation': float(max(0, -slack.min())),
            'min_variable': lower, 'max_upper_bound_violation': float(max(0, np.max(x)-upper_bound)), 'objective_recomputed_rub': float(c@x)}, slack

def write_csv(path, fields, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def solve_worker(name):
    dishes, config = load_data(); scenario=config['scenarios'][name]
    c,A,b,labels=build_model(dishes,config,scenario)
    res=linprog(c,A_ub=A,b_ub=b,bounds=(0,config['max_portions_per_dish_day']),
                method='highs-ds',options={'disp':True,'presolve':True,
                'primal_feasibility_tolerance':TOL,'dual_feasibility_tolerance':TOL})
    out={'scenario':name,'status':int(res.status),'message':res.message,'iterations':int(res.nit)}
    if res.status==0:
        check,slack=verify(np.asarray(res.x),c,A,b,config['max_portions_per_dish_day'])
        check['objective_solver_rub']=float(res.fun)
        check['objective_difference']=abs(float(res.fun)-check['objective_recomputed_rub'])
        assert check['passed'] and check['objective_difference']<=TOL
        out.update(check)
        solution=[]
        for day in range(config['days']):
            for j,d in enumerate(dishes):
                v=float(res.x[day*len(dishes)+j])
                solution.append({'day':day+1,'dish':d['name'],'portions':v,'cost_rub':v*float(d['cost_rub'])})
        write_csv(ROOT/f'results/{name}_solution.csv',list(solution[0]),solution)
        constraints=[{'constraint':label,'lhs':float(A[i]@res.x),'rhs':float(b[i]),
                      'slack':float(slack[i]),'active':bool(abs(slack[i])<=TOL),
                      'marginal_transformed_rhs':float(res.ineqlin.marginals[i])}
                     for i,label in enumerate(labels)]
        write_csv(ROOT/f'results/{name}_constraints.csv',list(constraints[0]),constraints)
        out['zero_variables']=int(np.sum(np.abs(res.x)<=TOL))
        # Локальная проверка каждой активной цены: малая положительная добавка к преобразованной правой части.
        sensitivity=[]
        for i,label in enumerate(labels):
            if abs(slack[i])>TOL: continue
            eps=0.001; changed=b.copy(); changed[i]+=eps
            r=linprog(c,A_ub=A,b_ub=changed,bounds=(0,config['max_portions_per_dish_day']),method='highs-ds')
            if r.success:
                sensitivity.append({'constraint':label,'rhs_delta':eps,'predicted_delta_rub':float(res.ineqlin.marginals[i]*eps),
                                    'actual_delta_rub':float(r.fun-res.fun)})
        if sensitivity: write_csv(ROOT/f'results/{name}_sensitivity.csv',list(sensitivity[0]),sensitivity)
        out['capacity_shadow_price']=float(res.ineqlin.marginals[-1])
    elif res.status==2:
        # Необходимое условие: сумма минимальных калорий <= ёмкость * максимум калорий на порцию.
        need=config['days']*config['daily_limits']['kcal'][0]
        available=scenario['portion_capacity']*max(float(d['kcal']) for d in dishes)
        out['infeasibility_certificate']={'required_kcal':need,'upper_bound_kcal':available,'contradiction':need>available}
    (ROOT/f'results/{name}_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

def run():
    _,config=load_data(); summaries=[]
    for name in config['scenarios']:
        with (ROOT/f'results/{name}_solver.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name],stdout=log,stderr=subprocess.STDOUT,check=True)
        summaries.append(json.loads((ROOT/f'results/{name}_verification.json').read_text()))
    fields=['scenario','status','objective_recomputed_rub','capacity_shadow_price','zero_variables','max_constraint_violation']
    write_csv(ROOT/'results/scenarios.csv',fields,[{k:r.get(k,'') for k in fields} for r in summaries])
    (ROOT/'results/environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'solver':'HiGHS, version in solver logs'},indent=2))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker': solve_worker(sys.argv[2])
    else: run()
