# Script for playing around with FF-based interpolation of ab initio BO-Surfaces
#
# AH 3/9/2014

# Load pyQChem for filehandling
import sys
#sys.path.append('/home/daniel/Dropbox/UNI - Kopie/Bach/butadien/pyQChem2')
import pyQChem2.start as qc
import numpy as np
import copy
#sys.path.append('c:/user/lukic/Dropbox/UNI - Kopie/BachelorArbeit/Butadien_New/pyQChem2')

#from pybrain.datasets import SupervisedDataSet

# ----------------- Load energies and corresponding geometries ----------------

aimd_out = qc.read("pyQChem2/aimd.out",silent=True)
#/Users/lukic/Dropbox/UNI - Kopie/BachelorArbeit/ButadienTens/ButadienTens/aimd_old.out
aimd_job2 = aimd_out.list_of_jobs[1]

trajectory_geometries =  aimd_job2.aimd.geometries
trajectory_energies =  aimd_job2.aimd.energies

# Some helper functions to transform the xyz geometries
def xyz2zmat(geometry):
    # Calculates 24 internal coordinates from the xyz geometry
    internal_coordinates = []
    internal_coordinates.append(calc_distance(geometry.xyzs[0], geometry.xyzs[1])) # distance of C1 to C2
    internal_coordinates.append(calc_distance(geometry.xyzs[1], geometry.xyzs[2])) # distance of C2 to C3
    internal_coordinates.append(calc_angle(geometry.xyzs[0], geometry.xyzs[1], geometry.xyzs[2])) # angle between C1, C2 and C3
    internal_coordinates.append(calc_distance(geometry.xyzs[2], geometry.xyzs[3])) # distance of C3 to C4
    internal_coordinates.append(calc_angle(geometry.xyzs[1], geometry.xyzs[2], geometry.xyzs[3])) # angle between C2, C3 and C4
    internal_coordinates.append(calc_dihedral(geometry.xyzs[0], geometry.xyzs[1], geometry.xyzs[2], geometry.xyzs[3])) # dihedral between C1, C2, C3 and C4

    internal_coordinates.append(calc_distance(geometry.xyzs[0], geometry.xyzs[4])) # distance of C1 to H1
    internal_coordinates.append(calc_angle(geometry.xyzs[4], geometry.xyzs[0], geometry.xyzs[1])) # angle between H1, C1 and C2
    internal_coordinates.append(calc_dihedral(geometry.xyzs[4], geometry.xyzs[0], geometry.xyzs[1], geometry.xyzs[2])) # dihedral between H1, C1, C2 and C3

    internal_coordinates.append(calc_distance(geometry.xyzs[0], geometry.xyzs[5])) # distance of C1 to H2
    internal_coordinates.append(calc_angle(geometry.xyzs[5], geometry.xyzs[0], geometry.xyzs[1])) # angle between H2, C1 and C2
    internal_coordinates.append(calc_dihedral(geometry.xyzs[5], geometry.xyzs[0], geometry.xyzs[1], geometry.xyzs[2])) # dihedral between H2, C1, C2 and C3

    internal_coordinates.append(calc_distance(geometry.xyzs[1], geometry.xyzs[6])) # distance of C2 to H3
    internal_coordinates.append(calc_angle(geometry.xyzs[6], geometry.xyzs[1], geometry.xyzs[0])) # angle between H3, C2 and C1
    internal_coordinates.append(calc_dihedral(geometry.xyzs[6], geometry.xyzs[1], geometry.xyzs[2], geometry.xyzs[3])) # dihedral between H3, C2, C3 and C4

    internal_coordinates.append(calc_distance(geometry.xyzs[2], geometry.xyzs[7])) # distance of C3 to H4
    internal_coordinates.append(calc_angle(geometry.xyzs[7], geometry.xyzs[2], geometry.xyzs[3])) # angle between H4, C3 and C4
    internal_coordinates.append(calc_dihedral(geometry.xyzs[7], geometry.xyzs[3], geometry.xyzs[2], geometry.xyzs[1])) # dihedral between H3, C3, C2 and C1

    internal_coordinates.append(calc_distance(geometry.xyzs[3], geometry.xyzs[8])) # distance of C4 to H5
    internal_coordinates.append(calc_angle(geometry.xyzs[8], geometry.xyzs[3], geometry.xyzs[2])) # angle between H5, C4 and C3
    internal_coordinates.append(calc_dihedral(geometry.xyzs[8], geometry.xyzs[3], geometry.xyzs[2], geometry.xyzs[1])) # dihedral between H5, C4, C3 and C2

    internal_coordinates.append(calc_distance(geometry.xyzs[3], geometry.xyzs[9])) # distance of C4 to H6
    internal_coordinates.append(calc_angle(geometry.xyzs[9], geometry.xyzs[3], geometry.xyzs[2])) # angle between H6, C4 and C3
    internal_coordinates.append(calc_dihedral(geometry.xyzs[9], geometry.xyzs[3], geometry.xyzs[2], geometry.xyzs[1])) # dihedral between H6, C4, C3 and C2
    return internal_coordinates

  
def calc_distance(a1,a2):
    # Calculates the distance between two atoms a1 and a2 
    return np.linalg.norm(a2-a1)

def calc_angle(a1,a2,a3):
    # Calculates the angle inbetween the vectors a1-a2 and a3-a2
    r1 = a1-a2
    r2 = a3-a2
    #return np.arccos(np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)))
    return np.arctan2(np.linalg.norm(np.cross(r1,r2)),np.dot(r1,r2))

def calc_dihedral(a1,a2,a3,a4):
    # Calculates the dihedral angle of a chain of four atoms
    r1 = a1-a2
    r2 = a3-a2
    r3 = a4-a3
    n1 = np.cross(r1,r2)
    n2 = np.cross(r2,r3)
    #return np.arccos(np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)))
    return np.arctan2(np.dot(np.cross(n1,n2),r2/np.linalg.norm(r2)),np.dot(n1,n2))

def normalize_data(zmat):
    result = copy.copy(zmat)
    opt = aimd_job2.aimd.geometries[0]
    dist_indices = [0, 1, 3, 6, 9, 12, 15, 18, 21]
    for i in dist_indices:
        result[i] = result[i]
    angle_indices = [2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23]
    for i in angle_indices:
        result[i] = np.cos(result[i])
    return result


def getData():
    data_intput = []
    data_target = []
    for geo, energy in zip(trajectory_geometries[1:],
                           trajectory_energies):
        data_intput.append(normalize_data(xyz2zmat(geo)))
        data_target.append(energy)
    return data_intput, data_target

data_in, data_tar = getData()
