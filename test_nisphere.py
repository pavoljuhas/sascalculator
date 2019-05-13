# coding: utf-8

"test SAS calculation for 30A diameter nickel sphere"

import pytest
import numpy

from sascalculator import SASCalculator

@pytest.fixture(scope='module')
def nisph20():
    from diffpy.structure import Lattice, loadStructure
    from diffpy.structure.tests.testutils import datafile
    from diffpy.structure.expansion.makeellipsoid import makeSphere
    ni = loadStructure(datafile('Ni.stru'), 'pdffit')
    sph = makeSphere(ni, 20 / 2.0)
    sph.placeInLattice(Lattice())
    return sph


def test_iqtot(nisph20):
    q0, iqtot0 = numpy.loadtxt('nisphere20.dat', unpack=True)
    sc = SASCalculator(rmax=22)
    q, iqtot = sc(nisph20)
    assert numpy.allclose(q0, q)
    assert numpy.allclose(numpy.log(iqtot0), numpy.log(iqtot))
    return
