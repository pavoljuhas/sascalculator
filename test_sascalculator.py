# coding: utf-8

"test SAS calculation for 30A diameter nickel sphere"

import pytest
import numpy

from sascalculator import SASCalculator
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

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


def test_fq(nisph20):
    sc = SASCalculator(rmax=22)
    dbpc = DebyePDFCalculator(rmax=sc.rmax, qmax=sc.qmax, qstep=sc.qstep)
    sc.eval(nisph20)
    dbpc.eval(nisph20)
    assert numpy.array_equal(dbpc.qgrid, sc.qgrid)
    assert numpy.allclose(dbpc.fq, sc.fq)
    return
