"""Create a table of tidal constituents
"""
import pathlib
import numpy as np
import pyTMD.arguments

# documentation directory
directory = pathlib.Path(__file__).parent

cons = {}
cons['sa'] = ('sa', 'Solar annual')
cons['ssa'] = ('ssa', 'Solar semiannual')
cons['mm'] = ('mm', 'Lunar monthly')
cons['msf'] = ('msf', 'Lunisolar synodic fortnightly')
cons['mf'] = ('mf', 'Lunar declinational fortnightly')
cons['mt'] = ('mt', 'Termensual')

cons['2q1'] = ('2q1', 'Smaller elliptical diurnal')
cons['sigma1'] = ('\u03C31', 'Lunar variational diurnal')
cons['q1'] = ('q1', 'Larger lunar elliptical diurnal')
cons['rho1'] = ('\u03C11', 'Larger lunar evectional diurnal')
cons['o1'] = ('o1', 'Lunar diurnal')
cons['tau1'] = ('\u03C41', '')
cons['m1'] = ('m1', 'Smaller lunar elliptical diurnal')
cons['chi1'] = ('\u03C71', 'Smaller evectional diurnal')
cons['pi1'] = ('\u03C01', 'Solar elliptical diurnal')
cons['p1'] = ('p1', 'Principal solar diurnal')
cons['s1'] = ('s1', 'Raditional solar diurnal')
cons['k1'] = ('k1', 'Principal declinational diurnal')
cons['psi1'] = ('\u1D2A1', 'Smaller solar elliptical diurnal')
cons['phi1'] = ('\u03C61', 'Second-order solar diurnal')
cons['theta1'] = ('\u03B81', 'Evectional diurnal')
cons['j1'] = ('j1', 'Smaller lunar elliptical diurnal')
cons['oo1'] = ('oo1', 'Second-order lunar diurnal')

cons['eps2'] = ('\u03F52', '')
cons['2n2'] = ('2n2', 'Second-order lunar elliptical semidiurnal')
cons['mu2'] = ('\u03BC2', 'Lunar variational')
cons['n2'] = ('n2', 'Larger lunar elliptical semidiurnal')
cons['nu2'] = ('\u03BD2', 'Larger lunar evectional semidiurnal')
cons['m2'] = ('m2', 'Principal lunar semidiurnal')
cons['lambda2'] = ('\u03BB2', 'Smaller lunar evectional')
cons['l2'] = ('l2', 'Smaller lunar elliptical semidiurnal')
cons['t2'] = ('t2', 'Larger solar elliptical semidiurnal')
cons['s2'] = ('s2', 'Principal solar semidiurnal')
cons['r2'] = ('r2', 'Smaller solar elliptical semidiurnal')
cons['k2'] = ('k2', 'Lunisolar declinational semidiurnal')
cons['eta2'] = ('\u03B72', '')
cons['m3'] = ('m3', 'Principal lunar terdiurnal')

# Cartwright and Edden (1973) table with updated values
table = pyTMD.arguments._ce1973_table_1
# read the table
CTE = pyTMD.arguments._parse_tide_potential_table(table)

# create constituents table
models_table = directory.joinpath('_assets', 'constituents.csv')
fid = models_table.open(mode='w', encoding='utf8')
# write to csv
print('Constituent,Doodson Number,Frequency (cpd),Description', file=fid)
for c,params in cons.items():
    DO = pyTMD.arguments.doodson_number(c).astype(str).zfill(7)
    i, = np.nonzero(CTE['DO'] == DO)
    amp = 100.0*np.abs(CTE['Hs3'][i])
    omega, = pyTMD.arguments.frequency(c)
    freq = 86400.0*omega/(2.0*np.pi)
    period = 1.0/freq
    print(f'{params[0]},{DO},{freq:0.8f},{params[1]}', file=fid)
fid.close()
