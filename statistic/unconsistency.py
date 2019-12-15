# -*- coding:utf-8 -*-
import numpy as np
from prettytable import PrettyTable

st1 = '18|21|22|23|22|22|19|19|22|24|24|24|22|20|20|23|24|25|24|23|20|21|24|24|25|24|23|19|19|23|24|24|24|23|21|18|21|21|21|21|20|18'
su1 = '3.42|3.41|3.42|3.42|3.41|3.42|3.43|3.44|3.44|3.41|3.41|3.41|3.41|3.42|3.42|3.43|3.44|3.45|3.41|3.41|3.41|3.41|3.42|3.43|3.44|3.45|3.44|3.42|3.41|3.41|3.42|3.42|3.42|3.43|3.44|3.45|3.41|3.41|3.41|3.41|3.42|3.42|3.43|3.44|3.45|3.41|3.41|3.41|3.41|3.42|3.43|3.43|3.45|3.45|3.42|3.41|3.42|3.42|3.42|3.43|3.43|3.45|3.44|3.41|3.41|3.41|3.41|3.42|3.43|3.43|3.44|3.44|3.41|3.41|3.41|3.41|3.42|3.43|3.44|3.45|3.44|3.41|3.41|3.41|3.41|3.42|3.43|3.43|3.44|3.45|3.40|3.41|3.41|3.41|3.42|3.42|3.43|3.44|3.44|3.41|3.41|3.41|3.41|3.42|3.43|3.44|3.45|3.44|3.42|3.42|3.42|3.42|3.43|3.43|3.44|3.45|3.44|3.42|3.42|3.42|3.42|3.42|3.42|3.44|3.45|3.44|3.42|3.41|3.41|3.42|3.42|3.43|3.44|3.45|3.45|3.41|3.42|3.41|3.42|3.42|3.43|3.44|3.45|3.45|3.41|3.42|3.42|3.43|3.42|3.44|3.44|3.45|3.44|3.41|3.42|3.42|3.43|3.42|3.43|3.44|3.45|3.44'
st2 = '29|35|37|36|34|35|32|28|30|30|30|31|32|31|37|38|35|33|33|34|33|37|38|37|38|40|42|39|33|36|36|36|36|35|34|35|36|36|36|36|36|35'
su2 = '3.45|3.44|3.44|3.44|3.44|3.45|3.46|3.46|3.48|3.43|3.44|3.43|3.45|3.46|3.46|3.48|3.47|3.47|3.44|3.44|3.44|3.44|3.45|3.46|3.48|3.47|3.49|3.46|3.45|3.45|3.46|3.46|3.47|3.48|3.47|3.48|3.44|3.45|3.44|3.45|3.46|3.47|3.47|3.46|3.48|3.44|3.45|3.44|3.44|3.46|3.47|3.47|3.47|3.48|3.45|3.44|3.42|3.42|3.43|3.44|3.43|3.44|3.44|3.41|3.41|3.41|3.42|3.43|3.44|3.45|3.45|3.44|3.41|3.41|3.41|3.41|3.42|3.44|3.44|3.44|3.44|3.41|3.41|3.41|3.41|3.42|3.43|3.45|3.44|3.43|3.40|3.41|3.41|3.41|3.43|3.43|3.45|3.44|3.45|3.40|3.41|3.41|3.41|3.42|3.43|3.43|3.44|3.44|3.42|3.42|3.42|3.42|3.43|3.44|3.45|3.45|3.44|3.42|3.42|3.41|3.42|3.42|3.43|3.45|3.44|3.45|3.42|3.41|3.40|3.42|3.43|3.44|3.45|3.43|3.45|3.42|3.42|3.41|3.42|3.43|3.45|3.44|3.44|3.45|3.42|3.42|3.42|3.44|3.43|3.45|3.45|3.45|3.45|3.42|3.42|3.43|3.43|3.44|3.45|3.44|3.45|3.45'

t1 = [int(x) for x in st1.split("|")]
u1 = [float(x) for x in su1.split("|")]
t2 = [int(x) for x in st2.split("|")]
u2 = [float(x) for x in su2.split("|")]

table = PrettyTable(["Date", "U_std", "Umax-Umin", "T_std", "Tmax-Tmin"])
table.add_row(["2018/01/08", np.std(u1), np.max(u1) - np.min(u1), np.std(t1), np.max(t1) - np.min(t1)])
table.add_row(["2018/12/10", np.std(u2), np.max(u2) - np.min(u2), np.std(t2), np.max(t2) - np.min(t2)])

print(table)
