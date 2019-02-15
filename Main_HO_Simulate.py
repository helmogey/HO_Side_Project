
from math import log10
import numpy as np


no_of_users = 100
delta_F = 15*1e3
Boltzmann = 1.38*10e-23
temprature = 290


R = 500
Cx = 5 * R
Cy = 5 * R
X = R * (((3) ** 0.5) / 2)
Y = R * 0.5

HM = 3
Alfa = 0.5
Beta = 0.5
G = [-1, 0, 1, 2, 5, 7]

Total_Macro_Loc = np.array([[Cx, Cy],[Cx + X, Y + R + Y],[Cx - X, Cy + R + Y],[Cx - 2 * X, Cy],[Cx - X, Cy - R - Y],
                   [Cx + X, Cy - R - Y],[Cx + (2 * X), Cy],[Cx + (2 * X), Cy + 2 * (Y + R)],
                   [Cx, (Cy + (2 * (Y + R)))],[(Cx - (2 * X)), (Cy + (2 * (Y + R)))],[(Cx - (3 * X)), (Cy + R + Y)],
                   [(Cx - (4 * X)), Cy],[(Cx - (3 * X)), (Cy - R - Y)],[(Cx - (2 * X)), (Cy - (2 * (R + Y)))],
                   [Cx, (Cy - (2 * (R + Y)))],[(Cx + (2 * X)), (Cy - (2 * (R + Y)))],[(Cx + (3 * X)), (Cy - R - Y)],
                   [(Cx + (4 * X)), Cy],[(Cx + (3 * X)), (Cy + R + Y)]])

Macro_Power = 27
Macro_G = 0
Macro_Freq = 2 * 10 ** 9
Macro_BW = 10 * 10 ** 6
Macro_No_Of_Time_Slots = 8
Macro_No_Of_Subchannels = 10
Macro_h = 32

Femto_numbers = 30
Femto_G = 0
Femto_Power = 15
Femto_h = 10
Femto_Loc = np.zeros(Femto_numbers, 2)



Macro_numbers = 19
Macro_Loc = Total_Macro_Loc[1:Macro_numbers,:]

Macro_Vertices  = np.zeros((7,2,Macro_numbers))

for i,(cx,cy) in enumerate(Macro_Loc):

    Macro_Vertices[:,0,i] = [ cx+X, cx, cx-X, cx-X, cx, cx+X, cx+X]
    Macro_Vertices[:,1,i] = [cy+Y, cy+R, cy + Y,cy - Y,cy - R,cy - Y,cy + Y]












