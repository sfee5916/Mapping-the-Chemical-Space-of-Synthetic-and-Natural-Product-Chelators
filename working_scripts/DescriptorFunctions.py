import rdkit
from rdkit import Chem
import numpy as np
import math

## function definition
def CalculateAPol(mol,includeImplicitHs=True): ## code from: https://gist.github.com/greglandrum/7936fcf631bfdae0041e298421554bec
    # original code uses alternative apol values. these have been adapted to fit those used in the 'polarizability94.txt' used in Mordred. A copy is available in 'input_data/'
    atomPols = [0.666793, 0.2050522, 24.33, 5.6, 3.03, 1.67, 1.1, 0.802, 0.557, 0.39432, 24.11, 10.6, 6.8, 5.53, 3.63, 2.9, 2.18, 1.6411, 43.06, 
                22.8, 17.8, 14.6, 12.4, 11.6, 9.4, 8.4, 7.5, 6.8, 6.2, 5.75, 8.12, 5.84, 4.31, 3.77, 3.05, 2.4844, 47.24, 23.5, 22.7, 17.9, 15.7, 
                12.8, 11.4, 9.6, 8.6, 4.8, 6.78, 7.36, 10.2, 7.84, 6.6, 5.5, 5.35, 4.044, 59.42, 39.7, 31.1, 29.6, 28.2, 31.4, 30.1, 28.8, 27.7, 
                23.5, 25.5, 24.5, 23.6, 22.7, 21.8, 20.9, 21.9, 16.2, 13.1, 11.1, 9.7, 8.5, 7.6, 6.5, 5.8, 5.02, 7.6, 7.01, 7.4, 6.8, 6.0, 5.3, 
                48.6, 38.3, 32.1, 32.1, 25.4, 24.9, 24.8, 24.5, 23.3, 23.0, 22.7, 20.5, 19.7, 23.8, 18.2, 16.4]    
    contribs = []
    res = 0.0
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum<=len(atomPols):
            apol = atomPols[anum]
            if includeImplicitHs:
                apol += atomPols[1] * atom.GetTotalNumHs(includeNeighbors=False)
            contribs.append(apol)
            res += apol
        else:
            raise ValueError(f"atomic number {anum} not found")
    # return res,contribs
    return res

def CalculateBPol(mol): ## code derived from mordred github: https://github.com/mordred-descriptor/mordred/blob/develop/mordred/Polarizability.py
    # uses 'polarizability94.txt' used in Mordred. A copy is available in 'input_data/'
    atomPols = [0.666793, 0.2050522, 24.33, 5.6, 3.03, 1.67, 1.1, 0.802, 0.557, 0.39432, 24.11, 10.6, 6.8, 5.53, 3.63, 2.9, 2.18, 1.6411, 43.06, 
                22.8, 17.8, 14.6, 12.4, 11.6, 9.4, 8.4, 7.5, 6.8, 6.2, 5.75, 8.12, 5.84, 4.31, 3.77, 3.05, 2.4844, 47.24, 23.5, 22.7, 17.9, 15.7, 
                12.8, 11.4, 9.6, 8.6, 4.8, 6.78, 7.36, 10.2, 7.84, 6.6, 5.5, 5.35, 4.044, 59.42, 39.7, 31.1, 29.6, 28.2, 31.4, 30.1, 28.8, 27.7, 
                23.5, 25.5, 24.5, 23.6, 22.7, 21.8, 20.9, 21.9, 16.2, 13.1, 11.1, 9.7, 8.5, 7.6, 6.5, 5.8, 5.02, 7.6, 7.01, 7.4, 6.8, 6.0, 5.3, 
                48.6, 38.3, 32.1, 32.1, 25.4, 24.9, 24.8, 24.5, 23.3, 23.0, 22.7, 20.5, 19.7, 23.8, 18.2, 16.4]
    def getBondPol(bond):
        a = bond.GetBeginAtom().GetAtomicNum()
        b = bond.GetEndAtom().GetAtomicNum()
        return abs(atomPols[a]-atomPols[b])
    return float(sum([getBondPol(bond) for bond in mol.GetBonds()]))

def num_S(mol):
    return sum([1 for a in mol.GetAtoms() if a.GetAtomicNum()==16])
def num_O(mol):
    return sum([1 for a in mol.GetAtoms() if a.GetAtomicNum()==8])
def num_N(mol):
    return sum([1 for a in mol.GetAtoms() if a.GetAtomicNum()==7])

def prop_S(mol):
    return sum([1 for a in mol.GetAtoms() if a.GetAtomicNum()==16])/len([a for a in mol.GetAtoms()])
def prop_O(mol):
    return sum([1 for a in mol.GetAtoms() if a.GetAtomicNum()==8])/len([a for a in mol.GetAtoms()])
def prop_N(mol):
    return sum([1 for a in mol.GetAtoms() if a.GetAtomicNum()==7])/len([a for a in mol.GetAtoms()])

def bottcher_complexity(InpMol):
    ## noted tha Bottcher treated the double bond in carboxyls as delocalised. A fair point but really annoying to implement with chemistry coding packages
    mol = Chem.RWMol(InpMol)
    CarboxylicAcid = Chem.MolFromSmarts('[#8D1]-[#6]=[#8X1]')
    [(mol.GetBondBetweenAtoms(cbx[0],cbx[1]).SetBondType(Chem.rdchem.BondType.ONEANDAHALF),mol.GetBondBetweenAtoms(cbx[1],cbx[2]).SetBondType(Chem.rdchem.BondType.ONEANDAHALF)) for cbx in mol.GetSubstructMatches(CarboxylicAcid)]
    Chem.SanitizeMol(mol)

    Rank = list(Chem.CanonicalRankAtoms(mol,breakTies=False))
    NonUniques = [any([rr==r for m,rr in enumerate(Rank) if m!=n]) for n,r in enumerate(Rank)]
    PT = Chem.GetPeriodicTable()
    V = [PT.GetNOuterElecs(a.GetAtomicNum()) for a in mol.GetAtoms()]
    B = [sum([b.GetBondTypeAsDouble() for b in a.GetBonds()]) for a in mol.GetAtoms()]

    def UniqueBondedNeighbours(atom):
        d,Seen,aI = 0,[],atom.GetIdx()
        for n in atom.GetNeighbors():
            nI = n.GetIdx()
            if V[nI]*B[nI]>1:
                Id = (Rank[nI],mol.GetBondBetweenAtoms(nI,aI).GetBondTypeAsDouble())
                if not Id in Seen:
                    Seen.append(Id)
                    d+=1
        return d

    D = [UniqueBondedNeighbours(a) for a in mol.GetAtoms()]
    E = [len(set([n.GetAtomicNum() for n in a.GetNeighbors()]+[a.GetAtomicNum()])) for a in mol.GetAtoms()]
    StereoAtoms = [(mol.GetBondWithIdx(i.centeredOn).GetEndAtomIdx(),mol.GetBondWithIdx(i.centeredOn).GetBeginAtomIdx()) if 'bond' in i.type.name else [i.centeredOn] if 'atom' else [] in i.type.name for i in Chem.FindPotentialStereo(mol)]
    StereoAtoms = [j for i in StereoAtoms for j in i]
    S = [2 if i.GetIdx() in StereoAtoms else 1 for i in mol.GetAtoms()]
    res = np.array([D,E,S,V,B]).T
    return sum([d*e*s*np.log2(v*b) for d,e,s,v,b in res])-0.5*sum([d*e*s*np.log2(v*b) for d,e,s,v,b in res[NonUniques]])

def get_prop_aromatic(mol):
    aro = len([b for b in mol.GetBonds() if b.GetIsAromatic()])
    return aro/len(mol.GetBonds())

def get_prop_ring(mol):
    aro = len([b for b in mol.GetBonds() if b.IsInRing()])
    return aro/len(mol.GetBonds())

## These descriptors were identified through Mordred. However mordred's implementation sets unneccesary errors (e.g. Boron cannot have 4 bonds)
def RotRatio(mol):
    rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    bs = len([b for b in mol.GetBonds()])
    return rot/bs