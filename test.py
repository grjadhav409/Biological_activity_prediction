import numpy as np
s_value = [34, 35]
p_value=[]
for i in s_value:
    p_value.append(np.around(- np.log10(i / (10 ** (9))), 2))


print(p_value)