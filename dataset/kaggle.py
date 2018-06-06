import qmmlpack as qmml
import numpy as np
import ase.io as asio
import csv
from cmlkit.dataset import Dataset, Subset
import cmlkit.indices as cmli

# The following assumes that you have downloaded the Kaggle dataset from
# https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data,
# in particular train.csv.zip and train.zip, and have unzipped them
# into this directory

# You also need to have installed ASE, like so
# conda install -c conda-forge ase

# Use ASE to read the input files, which are in
# aims format, despite having the xyz extension

z = []
r = []
b = []

for i in range(2400):
    f = 'train/{}/geometry.xyz'.format(i + 1)
    structure = asio.read(f, format='aims')
    z.append(structure.numbers)
    r.append(structure.positions)
    b.append(structure.cell)

# Convert to numpy arrays
z = np.array(z)
r = np.array(r)
b = np.array(b)

# Parse the CSV file for properties

sg = []
fe = []
bg = []

with open('train.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sg.append(int(row['spacegroup']))
        fe.append(float(row['formation_energy_ev_natom']))
        bg.append(float(row['bandgap_energy_ev']))

# Once again convert to numpy array
fe = np.array(fe)
sg = np.array(sg)
bg = np.array(bg)

# The following computes some associated quantities

n_atoms = np.array([len(zz) for zz in z])  # count atoms in unit cell
n_sub = np.array([len(zz[zz != 8]) for zz in z])  # count atoms that are not O (the number of substitution sites)

fepa = fe * n_sub / n_atoms  # formation energy per atom
fecreal = fepa * n_atoms  # formation energy per unit cell (NOT per site)

# And now we actually create the Dataset

desc = "Dataset (training portion) from the NOMAD2018 Kaggle challenge: \
Relaxed geometries and their properties. Note that fe is not the formation energy per atom but per substitution site!"

data = Dataset('kaggle',
               z, r, b,
               {'fe': fe, 'fepa': fepa, 'fecreal': fecreal, 'n_atoms': n_atoms, 'n_sub': n_sub, 'sg': sg},
               desc=desc, family='tco')

# And save. Bam!
data.save()

# Let's now create a model building and validation split

np.random.seed(2312)
rest, build = cmli.twoway_split(data.n, 2000)

sub1 = Subset(data, build, name='build', desc='Randomly picked subset of 2000 structures')
sub2 = Subset(data, rest, name='rest', desc='Randomly picked subset of 400 remaining structures (complement to build)')
sub1.save()
sub2.save()
