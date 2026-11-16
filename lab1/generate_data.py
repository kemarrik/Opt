"""Воспроизводимые учебные данные; значения не являются справочником питания."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(10006)
    # Имя, белки, жиры, углеводы на условную порцию; ккал = 4Б+9Ж+4У.
    base=[('Овсяная каша',8,7,42),('Гречневая каша',7,4,40),('Рис',5,3,50),
          ('Макароны',8,4,48),('Куриная грудка',30,5,0),('Рыба',25,8,0),
          ('Говядина',24,15,0),('Омлет',14,16,3),('Творог',25,7,5),
          ('Чечевица',16,3,35),('Овощной салат',3,8,12),('Овощной суп',4,5,20),
          ('Хлеб',5,2,30),('Йогурт',7,4,15),('Фруктовый салат',1,1,30)]
    prices=[28,25,22,24,95,100,120,55,65,35,30,32,15,35,40]
    with (ROOT/'data/dishes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['name','protein','fat','carbs','kcal','cost_rub'])
        for (name,p,fat,c),price in zip(base,prices):
            p,fat,c=[round(v*rng.uniform(.95,1.05),2) for v in (p,fat,c)]
            w.writerow([name,p,fat,c,round(4*p+9*fat+4*c,2),round(price*rng.uniform(.9,1.1),2)])
    config={'variant':9,'N':6,'Q':0,'Y':26,'seed':10006,'days':5,'max_portions_per_dish_day':3,
            'daily_limits':{'kcal':[1700,2100],'protein':[80,130],'fat':[45,80],'carbs':[200,300]},
            'scenarios':{'base':{'portion_capacity':42},'deficit':{'portion_capacity':33.6},'stress':{'portion_capacity':25.2}}}
    (ROOT/'data/config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
