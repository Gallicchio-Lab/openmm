import unittest
from openmm import *
from openmm.app import *
from openmm.unit import *

class TestATMForce(unittest.TestCase):
    """Tests the ATMForce"""

    def test2ParticlesNonbonded(self):
        """Test for a Nonbonded force previously added to the System"""
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)

        nbforce = NonbondedForce();
        nbforce.addParticle( 1.0, 1.0, 1.0)
        nbforce.addParticle(-1.0, 1.0, 1.0)

        system.addForce(nbforce)

        atmforce = ATMForce(0.5, 0.5, 0, 0, 0,   0, 0, 0,  1.0)
        atmforce.addParticle()
        p = atmforce.addParticle()
        atmforce.setParticleTransformation(p, FixedDisplacement(Vec3(1., 0., 0.)))

        atmforce.addForce(copy.copy(nbforce))
        system.removeForce(0)
        system.addForce(atmforce)

        integrator = VerletIntegrator(1.0)
        platform = Platform.getPlatform('Reference')
        context = Context(system, integrator, platform)

        positions = []
        positions.append(Vec3(0., 0., 0.))
        positions.append(Vec3(1., 0., 0.))
        context.setPositions(positions)

        state = context.getState(getEnergy = True, getForces = True)
        epot = state.getPotentialEnergy()
        
        (u1, u0, energy) = atmforce.getPerturbationEnergy(context)
        epert = u1 - u0

        #print("Potential energy = ", epot)
        #print("ATM perturbation energy = ", epert)
        
        epot_expected = -104.2320*kilojoules_per_mole
        epert_expected = 69.4062*kilojoules_per_mole
        assert( abs(epot-epot_expected) < 1.e-3*kilojoules_per_mole )
        assert( abs(epert-epert_expected) < 1.e-3*kilojoules_per_mole )
        assert isinstance(atmforce.getParticleTransformation(0), FixedDisplacement)

    def test3ParticlesNonbondedSwap(self):
        """Test coordinate swap"""
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)

        nbforce = NonbondedForce();
        nbforce.addParticle( 0.0, 1.0, 1.0)
        nbforce.addParticle( 1.0, 1.0, 1.0)
        nbforce.addParticle(-1.0, 1.0, 1.0)

        atmforce = ATMForce(0.5, 0.5, 0, 0, 0,   0, 0, 0,  1.0)
        atmforce.addParticle() #particle 0 is not displaced

        #particle 1's coordinate is swapped with 2
        p = atmforce.addParticle()
        atmforce.setParticleTransformation(p, ParticleOffsetDisplacement(2,  1))

        #particle 2's coordinate is swapped with 1
        p = atmforce.addParticle()
        atmforce.setParticleTransformation(p, ParticleOffsetDisplacement(1,  2))

        atmforce.addForce(nbforce)
        system.addForce(atmforce)

        integrator = VerletIntegrator(1.0)
        platform = Platform.getPlatform('Reference')
        context = Context(system, integrator, platform)

        positions = []
        positions.append(Vec3( 0., 0., 0.))
        positions.append(Vec3( 1., 0., 0.))
        positions.append(Vec3(-1., 0., 0.))
        context.setPositions(positions)

        state = context.getState(getEnergy = True, getForces = True)
        epot = state.getPotentialEnergy()

        (u1, u0, energy) = atmforce.getPerturbationEnergy(context)
        epert = u1 - u0

        #print("Potential energy = ", epot)
        #print("ATM perturbation energy = ", epert)

        epot_expected = -69.52925*kilojoules_per_mole
        epert_expected = 0*kilojoules_per_mole
        assert( abs(epot-epot_expected) < 1.e-3*kilojoules_per_mole )
        assert( abs(epert-epert_expected) < 1.e-3*kilojoules_per_mole )

    def testFrameDisplacement(self):
        """Test frame-based displacements"""
        system = System()
        # source frame particles: particles 0, 1, and 2
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)
        # destination frame particles: particles 3, 4, and 5
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)
        # particle on origin frame: particle 6
        system.addParticle(1.0)
        # particle on destination frame: particle 7
        system.addParticle(1.0)
        
        frame0 = [0, 1, 2]
        frame1 = [3, 4, 5]
        side0 = 6
        side1 = 7

        displ = Vec3(0, 5, 0)
        positions = []
        positions.append(Vec3( 0., 0., 0.)) #A
        positions.append(Vec3( 0., 0., 2.)) #B
        positions.append(Vec3( 1., 0., 1.)) #C
        positions.append(Vec3( 0., 0., 0.) + displ) #A'
        positions.append(Vec3( 2., 0., 0.) + displ) #B'
        positions.append(Vec3( 2., 0.,-2.) + displ) #C'
        positions.append(Vec3(-1., 1., 1.)) #D, side0
        positions.append(Vec3(-3., 1., 3.) + displ) #E', side1

        bondforce = CustomBondForce("1.0*r^2")
        bondforce.addBond(side0, frame0[2]); #bond between D and C
        bondforce.addBond(side1, frame1[2]); #bond between E' and C'

        atmforce = ATMForce(0.5, 0.5, 0, 0, 0,   0, 0, 0,  1.0)
        for i in range(8):
            atmforce.addParticle()
        #swap frame particles
        atmforce.setParticleTransformation(0, ParticleOffsetDisplacement(frame1[0],  frame0[0])) #A->A'
        atmforce.setParticleTransformation(1, ParticleOffsetDisplacement(frame1[1],  frame0[1])) #B->B'
        atmforce.setParticleTransformation(2, ParticleOffsetDisplacement(frame1[2],  frame0[2])) #C->C'
        atmforce.setParticleTransformation(3, ParticleOffsetDisplacement(frame0[0],  frame1[0])) #A'->A
        atmforce.setParticleTransformation(4, ParticleOffsetDisplacement(frame0[1],  frame1[1])) #B'->B
        atmforce.setParticleTransformation(5, ParticleOffsetDisplacement(frame0[2],  frame1[2])) #C'->C
        #swap frames of sidechain particles
        atmforce.setParticleTransformation(6, ParticleFrameDisplacement(frame1[0], frame1[1], frame1[2],  frame0[0], frame0[1], frame0[2])) #D->D'
        atmforce.setParticleTransformation(7, ParticleFrameDisplacement(frame0[0], frame0[1], frame0[2],  frame1[0], frame1[1], frame1[2])) #E'->E
        
        atmforce.addForce(bondforce)
        system.addForce(atmforce)

        integrator = VerletIntegrator(1.0)
        platform = Platform.getPlatform('Reference')
        context = Context(system, integrator, platform)
        context.setPositions(positions)

        state = context.getState(getEnergy = True, getForces = True)
        epot = state.getPotentialEnergy()

        (u1, u0, energy) = atmforce.getPerturbationEnergy(context)
        epert = u1 - u0

        #print("u0 = ", u0)
        #print("u1 = ", u1)

        u0_expected = 56.0*kilojoules_per_mole
        u1_expected = 44.0*kilojoules_per_mole
        assert( abs(u0 - u0_expected) < 1.e-3*kilojoules_per_mole )
        assert( abs(u1 - u1_expected) < 1.e-3*kilojoules_per_mole )
