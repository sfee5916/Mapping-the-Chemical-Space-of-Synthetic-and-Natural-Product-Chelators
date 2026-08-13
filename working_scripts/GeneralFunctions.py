import rdkit
from rdkit import Chem
from rdkit.Chem import MolStandardize
import itertools

def RDKitCarefulSanitize(mol):
    errors = []
    for flg in [rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL,rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_PROPERTIES,rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE]: ## trying all the sanitization flags, in previous runs: 1750 failed the "all", 1477 failed the "properties", and 92 failed the "kekulize"
        try:
            Chem.SanitizeMol(mol,flg)
            break
        except Exception as e:
            errors.append(str(e))
    return mol,errors

def CCDC_localise(mol):
    DelocalisedBonds = [b for b in mol.bonds if b.bond_type==7]
    if DelocalisedBonds:
        DelocalisedBondSets = []
        first = True
        while DelocalisedBonds:
            if first:
                DelocalisedBondSets.append([DelocalisedBonds[0]])
                DelocalisedBonds.remove(DelocalisedBonds[0])
                first = False
            else:
                res = [(DelocalisedBonds.remove(b),DelocalisedBondSets[-1].append(b))  for a in DelocalisedBondSets[-1][-1].atoms for b in a.bonds if b in DelocalisedBonds] 
                if not res:
                    DelocalisedBondSets.append([DelocalisedBonds[0]])
                    DelocalisedBonds.remove(DelocalisedBonds[0])

        for DelocalisedBonds in DelocalisedBondSets:
                N = len(DelocalisedBonds)
                if N > 12:
                    return 12 ## calculation of permutations of > 12 delocalised electrons takes too long using this method. Number excluded is negligable
                Singles = int(N/2)
                Doubles = N-Singles
                Options = set(itertools.permutations([1]*Singles+[2]*Doubles))
                AdjacencyMat = [[any([b2 in n.bonds for n in b1.atoms]) for b2 in DelocalisedBonds ] for b1 in DelocalisedBonds]
                NO = []
                for n in range(N):
                    for m in range(N):
                        if m>n and AdjacencyMat[n][m]:
                            for o in Options:
                                if o[m] == o[n]:
                                    NO.append(o)
                ValidOptions = [o for o in Options if not o in NO]
                if ValidOptions:
                    for b,bt in zip(DelocalisedBonds,ValidOptions[0]):
                        b.bond_type = bt
                else:
                    return False
    return True