# coding: utf-8

"Show SAS simulation for 20A diameter nickel sphere"

import numpy
from matplotlib import pyplot as plt

from sascalculator import SASCalculator
from diffpy.structure import Lattice, loadStructure
from diffpy.structure.tests.testutils import datafile
from diffpy.structure.expansion.makeellipsoid import makeSphere

ni = loadStructure(datafile('Ni.stru'), 'pdffit')
sph = makeSphere(ni, 20 / 2.0)
sph.placeInLattice(Lattice())

sc = SASCalculator(qstep=.01, rmax=20)
q, iqtot = sc(sph)

_, ax = plt.subplots()
ax.loglog(q, iqtot)
ax.set(title='SAS intensity from nickel sphere, d=20A',
       xlabel='Q (1/A)', ylabel='total intensity')
plt.show()
