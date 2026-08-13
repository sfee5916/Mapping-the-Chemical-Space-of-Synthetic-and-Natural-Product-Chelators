## imports
import os
import csv
import math
import pickle
import multiprocessing as mp
from sklearn import preprocessing
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from ccdc.io import MoleculeReader

from working_scripts.GeneralFunctions import RDKitCarefulSanitize,CCDC_localise
import working_scripts.DescriptorFunctions as DescFn

from Bio import PDB
from Bio.PDB import PDBParser, PDBList
from rcsbapi.search  import search_attributes as attrs
##### Below removes error output messages. When dealing with large numbers of data these are expected and are hidden here to avoid filling up the terminal
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
##### TEST


## Function definitons
def CanoniseSMILES(smi,stereo=True):
    try:
        Mol = Chem.MolFromSmiles(smi)
    except:
        Mol = None
    if not Mol:
            Mol = Chem.MolFromSmiles(smi,sanitize=False)
    if not stereo:
        Chem.RemoveStereochemistry(Mol)
    if Mol:
        return Chem.MolToSmiles(Mol)
    else:
        return None
def StandardiseMolecules(mols):
    UCrg = rdMolStandardize.Uncharger()
    OutMols = []
    for M in tqdm(mols):
        OM = UCrg.uncharge(M)
        try:
            OM = OM
        except:
            None
        OutMols.append(OM)
    return OutMols

def GetCarboraneLikes(mol):
    RI = mol.GetRingInfo()
    for R in RI.AtomRings():
        if len(R) == 3:
            test = []
            for i in R:
                a = mol.GetAtomWithIdx(i)
                test.append(a.GetSymbol() in ['C','B','Si'] and any([n.IsInRingSize(3) and n.GetSymbol() in ['C','B','Si'] and not n.GetIdx() in R for n in a.GetNeighbors()]))
            if all(test):
                return True
    return False
def GetExplicitBondOrder(atom):
    BO = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])+atom.GetNumExplicitHs()
    if atom.GetSymbol() in ['B','H']:
        BO +=atom.GetFormalCharge()
    else:
        BO -=atom.GetFormalCharge()
    return BO
def SplitOnHs(mol):
    Omol = Chem.RWMol(mol)
    DoubleHs = [a for a in Omol.GetAtoms() if a.GetSymbol()=='H' and GetExplicitBondOrder(a)>1.0]
    Omol.BeginBatchEdit()
    [Omol.RemoveAtom(h.GetIdx()) for h in DoubleHs]
    Omol.CommitBatchEdit()
    return Omol
def Generate_CSD_data(CSDpath):
    with open(CSDpath,'r') as f:
        CSDEntries = [i.rstrip() for i in f]
    elements_to_exclude = alkali_metals+alkaline_earth_metals+lanthanides+actinides+post_transition_metals

    opt = rdMolStandardize.MetalDisconnectorOptions()
    opt.adjustCharges = True
    opt.removeHapticDummies = True
    opt.splitAromaticC = False
    opt.splitGrignards = False
    dcon = rdMolStandardize.MetalDisconnector(options=opt)
    SplitSMARTS = Chem.MolFromSmarts('[#'+',#'.join([str(i) for i in transition_metals+lanthanides+alkaline_earth_metals+post_transition_metals])+']~[#7,#8,#15,#16]')
    dcon.SetMetalNof(SplitSMARTS)
    opt.adjustCharges = False
    dconNC = rdMolStandardize.MetalDisconnector(options=opt)
    dconNC.SetMetalNof(SplitSMARTS)
    
    csd_mol_reader = MoleculeReader('CSD')
    results = []
    for CSDId in tqdm(CSDEntries):
        mm = csd_mol_reader.molecule(CSDId)
        for c in mm.components:
            Valid = CCDC_localise(c)
            if Valid and c.atoms:
                LigSmis,LigMols,LigMets,LigCoord = [],[],[],[]
                c.assign_bond_types(which='unknown')
                try:
                    mol = Chem.MolFromSmiles(c.to_string('smiles'),sanitize=False)
                    if not mol:
                        raise TypeError('no molecule generated') 
                except:
                    mol = Chem.MolFromMolBlock(c.to_string('mol'),sanitize=False)
                    if not mol:
                        continue
                mol,e = RDKitCarefulSanitize(mol)
                if sum([a.GetAtomicNum() in transition_metals for a in mol.GetAtoms()]) != 1:
                    continue
                MetalNeighbours = [(neighbour.GetIdx(),atom.GetSymbol()) for atom in mol.GetAtoms() if atom.GetAtomicNum() in transition_metals for neighbour in atom.GetNeighbors()]
                try:
                    SplitMol = dcon.Disconnect(mol)
                except:
                    SplitMol = dconNC.Disconnect(mol)
                if any([a.GetSymbol()=='H' and GetExplicitBondOrder(a)>1.0 for a in SplitMol.GetAtoms()]):
                    SplitMol = SplitOnHs(SplitMol)
                for fragMol,fragIdx in zip(Chem.GetMolFrags(SplitMol,asMols=True,sanitizeFrags=False),Chem.GetMolFrags(SplitMol,asMols=False)):
                    fragMol,e = RDKitCarefulSanitize(fragMol)
                    if any([atom.GetAtomicNum() in elements_to_exclude for atom in fragMol.GetAtoms()]):
                        continue
                    if any([b==Chem.rdchem.BondType.UNSPECIFIED for b in fragMol.GetBonds()]):
                        continue
                    FragMetalNeighbours = [(i,M) for i in fragIdx for j,M in MetalNeighbours if i==j]
                    if FragMetalNeighbours:
                        FragMetalNeighbours,Metals = list(zip(*FragMetalNeighbours))
                        if len(FragMetalNeighbours) <= 1:
                            continue
                        if any([mol.GetAtomWithIdx(i).GetAtomicNum() in [6,34,5,14,32,33,51,52] for i in FragMetalNeighbours]):
                            continue
                        if any([neighbour.GetAtomicNum() in [6,34,5,14,32,33,51,52] for i in FragMetalNeighbours if mol.GetAtomWithIdx(i).GetAtomicNum()==1 for neighbour in mol.GetAtomWithIdx(i).GetNeighbors()]):
                            continue
                        if GetCarboraneLikes(fragMol):
                            continue
                        if not any([a.GetAtomicNum()==6 for a in fragMol.GetAtoms()]):
                            continue
                        LigSmis.append(Chem.MolToSmiles(fragMol))
                        LigMols.append(fragMol)
                        LigMets.append(list(set(Metals)))
                        LigCoord.append(len(FragMetalNeighbours))
                for S,M,Me,C in zip(LigSmis,LigMols,LigMets,LigCoord):
                    results.append([CSDId,M,S,Me,C])
    CSDMols,CSDIds,CSDSmiles,CSDMets,CSDCoord = [],[],[],[],[]
    for c,m,s,mets,coord in results:
        if not mets:
            print('NO MET',c,m,s,mets)
        CSDMols.append(m)
        CSDIds.append(c)
        CSDMets.append(mets)
        CSDCoord.append(coord)
    CSDMols = StandardiseMolecules(CSDMols)
    CSDSmiles = [Chem.MolToSmiles(i) for i in CSDMols]
    CSDLigs = {k:[] for k in ['smiles','molecule','database ID','metal','denticity']}
    for S in tqdm(set(CSDSmiles)):
        CSDLigs['smiles'].append(S) 
        molF,refs,mets,coords = False,[],[],[]
        for smi,mol,c,met,coord in zip(CSDSmiles,CSDMols,CSDIds,CSDMets,CSDCoord):
            if S == smi:
                if not molF:
                    molF = True
                    CSDLigs['molecule'].append(mol)
                refs.append(c)
                mets+=met
                coords+=[str(coord)]
        CSDLigs['database ID'].append(';'.join(refs))
        if not mets:
            print('No Mets:',refs,mets)

        CSDLigs['metal'].append(';'.join(mets))
        CSDLigs['denticity'].append(';'.join(coords))
    return CSDLigs
        
def GetPDBStruct(PDB_ID,tries=10):
    pdbl = PDBList(verbose=False,server='https://files.wwpdb.org') # original code uses ftp server that is no longer supported :(
    pdbparser = PDBParser(QUIET=True)
    mmcifparser = PDB.MMCIFParser(QUIET=True)
    tried = 0
    while tried < tries: ## prevents an infinite loop if the entry cannot be found
        try:
            pdb_name = pdbl.retrieve_pdb_file(PDB_ID,pdir='output_data/temp',file_format='mmCif') 
            if os.path.isfile(pdb_name):
                    struct = mmcifparser.get_structure(PDB_ID, pdb_name)
            else:
                pdb_name = pdbl.retrieve_pdb_file(PDB_ID,pdir='output_data/temp',file_format='pdb')
                if os.path.isfile(pdb_name):
                    struct = pdbparser.get_structure(PDB_ID, pdb_name)
            return struct
        except Exception as e:
            print(e)
            tried +=1
    return None
def GetInfoFromResCodeAndPDBId(inp):
    pdb_id,res_code = inp
    struct = GetPDBStruct(pdb_id)
    info = []
    if struct:
        Residues = [i for i in struct.get_residues() if i.resname ==res_code ]
        metals = []; others = []
        for Res in Residues:
            if any([a.element == 'C' for a in Res.get_atoms()]):
                for A in Res.get_atoms():
                    if A.element in TransitionMetalSymbols:
                        metals.append(A)
                    else:
                        others.append(A)
        ProteinCoords = [a.get_coord() for i in struct.get_residues() if i.get_full_id()[3][0] ==' ' for a in i.get_atoms() if a.element != 'C' ]
        protein_bound = False
        for j,metal in enumerate(metals):
            MetalPos = metal.get_coord()
            coordination = 0
            ## testing if the ligand chelates the metal
            for j,A in enumerate(others):
                if A.element != 'C':
                    APos = A.get_coord()
                    coord_diff = math.sqrt(sum([(x-y)**2 for x,y in zip(APos,MetalPos)]))       
                    if coord_diff <= 3: ## angstrom distance to define metal coordinating atom
                        coordination +=1
            if coordination>2:
                ## testing the metal is bound to the protein
                for protein_coord in ProteinCoords:
                    coord_diff = math.sqrt(sum([(x-y)**2 for x,y in zip(MetalPos,protein_coord)]))       
                    if coord_diff <= 3: ## angstrom distance to define metal coordinating atom
                        protein_bound = True
                        break
                if protein_bound:
                    smi = PDBLigInfo[res_code][0]
                    info.append([res_code,smi,pdb_id,metal.element,coordination])
    return info
def GetInfoFromPDBId(pdb_id):
    global PDBLigInfo
    struct = GetPDBStruct(pdb_id)
    info = []
    if struct:
        HETGroups = [i for i in struct.get_residues() if i.get_full_id()[3][0] !=' ' and i.resname !='HOH' ]
        Metals = []; Ligands = []
        for RES in HETGroups:
            RESLen = len(list(RES.get_atoms()))==1
            if RESLen==1 and any([i.element.upper() in TransitionMetalSymbolsUpper for i in RES.get_atoms()]):
                Metals.append(list(RES.get_atoms())[0])
            elif any([i.element.upper()=='C' for i in RES.get_atoms()]): ## only ligands with C, excludes some obvious ones
                Ligands.append(RES)
        for met in Metals:
            met_coord = met.get_coord()
            for Ligand in Ligands:
                coordination = 0
                res_code = Ligand.get_resname()
                for A in Ligand.get_atoms():
                    if not A.element.upper() in ExcludedSymbolsUpper:
                        at_coords = A.get_coord()
                        MetalDistance = math.sqrt(sum([(x-y)**2 for x,y in zip(at_coords,met_coord)]))       
                        if MetalDistance <= 3: ## angstrom distance to define metal coordinating atom
                            coordination +=1
                if coordination:
                    if res_code in PDBLigInfo:
                        smi = PDBLigInfo[res_code][0]
                        info.append([res_code,smi,pdb_id,met.element,coordination])
    return info
def Generate_PDB_ligands():
    # Test 1: Transition metals and Ligands in different Het groups
    results_metal = []
    for metal_symbol in tqdm(TransitionMetalSymbolsUpper,desc='Finding PDB entries (1/2)'):
        q1 = attrs.rcsb_chem_comp_container_identifiers.comp_id == metal_symbol
        q2 = attrs.rcsb_entry_info.selected_polymer_entity_types == 'Protein (only)' # options Nucleic acid (only), Oligosaccharide (only), Other, Protein (only), Protein/NA, Protein/Oligosaccharide
        q3 = attrs.rcsb_nonpolymer_instance_annotation.type == 'HAS_METAL_COORDINATION_LINKAGE'
        query = q1 & q2 & q3
        result = query()
        results_metal +=result
    results_metal = list(set(results_metal))
    with mp.Pool(PROCS) as P:
        Test1Ligands = list(tqdm(P.imap(GetInfoFromPDBId,results_metal),total=len(results_metal),desc='PDB to ligand (1/2)'))
    Test1Ligands = [j for i in Test1Ligands for j in i if j]
    ## Test 2: Transition metals and Ligands in the same Het group
    extra_found = {k:v for k,v in PDBLigInfo.items() if any(["["+tm+"]" in v[0] for tm in TransitionMetalSymbols])}
    results_metal = []
    for k in tqdm(list(extra_found.keys()),desc='Finding PDB entried (2/2)'):
        ## test if this ligand is bound to some PDB entry
        q1 = attrs.rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id == k
        q2 = attrs.rcsb_entry_info.selected_polymer_entity_types == 'Protein (only)' # options Nucleic acid (only), Oligosaccharide (only), Other, Protein (only), Protein/NA, Protein/Oligosaccharide
        q3 = attrs.rcsb_nonpolymer_instance_annotation.type == 'HAS_METAL_COORDINATION_LINKAGE'
        query = q1 & q2 & q3
        result = list(query())
        output = [[id,k] for id in result]
        results_metal += output
    results_metal = [list(item) for item in set(tuple(row) for row in results_metal if row)]
    with mp.Pool(PROCS) as P:
        Test2Ligands = list(tqdm(P.imap(GetInfoFromResCodeAndPDBId,results_metal),total=len(results_metal),desc='PDB to ligand (2/2)'))
    Test2Ligands = [j for i in Test2Ligands for j in i if j]
    ## converting PDB entries into RDKit format
    res = []
    for RC,smi,ID,Met,Coord in Test1Ligands+Test2Ligands:
        m = Chem.MolFromSmiles(smi,sanitize=False)
        # an alternative metal removal method is applied since the one used in the CSD molecules did not remove some bonds to Hg
        AtomsToRemove = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() in transition_metals]
        if AtomsToRemove:
            rwmol = Chem.RWMol(m)
            rwmol.BeginBatchEdit()
            for ai in AtomsToRemove:
                rwmol.RemoveAtom(ai)
            rwmol.CommitBatchEdit()
            if rwmol:
                M = [rwm for rwm in Chem.GetMolFrags(rwmol,asMols=True,sanitizeFrags=False) if Chem.MolToSmiles(rwm) and  any([a.GetAtomicNum()==6 for a in rwm.GetAtoms()])]
                MLen = [len(m.GetAtoms()) for m in M ]
                ## assumes that the largest ligand bound to the transition metal is the ligand of interest. Here we are assuming all other ligands are water or similar.
                NewSmi = Chem.MolToSmiles(M[MLen.index(max(MLen))])
            else:
                NewSmi = None
        else:
            NewSmi = smi
        if NewSmi:
            if len(Met)>1:
                Met = Met[0]+Met[1].lower()
            res.append((NewSmi,ID,Met,Coord))

    mols = [Chem.MolFromSmiles(i[0],sanitize=False) for i in res]
    mols = [RDKitCarefulSanitize(i)[0] for i in mols]
    mols = StandardiseMolecules(mols)

    data = {}
    for s,(oldSmi,ID,Met,Coord) in zip([Chem.MolToSmiles(i) for i in mols],res):
        if not s in data:
            data[s] = [[ID],[Met],[str(Coord)]]
        else:
            data[s] = [i+j for i,j in zip(data[s],[[ID],[Met],[str(Coord)]])]

    res = {k:[] for k in ['smiles','molecule','database ID','metal','denticity']}
    for s,(IDs,Mets,Coords) in data.items():
        if not s in res['smiles']:
            res['smiles'].append(s)
            res['molecule'].append(RDKitCarefulSanitize(Chem.MolFromSmiles(s,sanitize=False))[0])
            res['database ID'].append(';'.join(IDs))
            res['metal'].append(';'.join(Mets))
            res['denticity'].append(';'.join(Coords))
    return res

def Generate_NP_data(COCONUTpath,LOTUSpath,SUPERNATURALpath):
    with open(COCONUTpath,'r', encoding="utf8") as f:
        ls = [i[:i.find(',"InChI')].split(',') if ',"InChI' in i else i[:i.find(',InChI')].split(',') for i in f][1:]
    coco_id = [i[0] for i in ls]
    coco_smi = [CanoniseSMILES(i[1]) for i in tqdm(ls)]
    coco_smi,coco_id = zip(*[(a,b) for a,b in zip(coco_smi,coco_id) if a])

    with open(LOTUSpath,'r') as f:
        ls = [i.rstrip().split('\t') for i in f]
    lotus_id = [i[1] for i in ls]
    lotus_smi = [CanoniseSMILES(i[0]) for i in tqdm(ls)]
    lotus_smi,lotus_id = zip(*[(a,b) for a,b in zip(lotus_smi,lotus_id) if a])

    with open(SUPERNATURALpath,'r', encoding="utf8") as f:
            ls = [i for i in csv.reader(f, quotechar='"', delimiter=';')][1:]
    ls = [i for i in ls if i[0] and i[5] and not i[5] == 'NA']
    supernatural_id = [i[0] for i in ls]
    supernatural_smi = [CanoniseSMILES(i[5]) for i in ls]
    supernatural_smi,supernatural_id = zip(*[(a,b) for a,b in zip(supernatural_smi,supernatural_id) if a])

    data = {k:[] for k in ['smiles','molecule','source','NP ID']}
    for smi,id,source in tqdm([ (coco_smi,coco_id,'COCONUT'),
                                (lotus_smi,lotus_id,'LOTUS'),
                                (supernatural_smi,supernatural_id,'Supernatural 3.0')
                            ],desc='Preparing molecules'):
        for s,I in zip(smi,id):
            mol = Chem.MolFromSmiles(s,sanitize=False)
            mol,e = RDKitCarefulSanitize(mol)
            data['molecule'].append(mol)
            data['source'].append(source)
            data['NP ID'].append(I)
    data['molecule'] = StandardiseMolecules(data['molecule'])
    data['smiles'] = [Chem.MolToSmiles(i) for i in data['molecule']]
    return data

def Generate_Siderite_data(path):
    InpData = pd.read_csv(path)
    data = {}
    data['molecule']   = [Chem.MolFromSmiles(i,sanitize=False) for i in InpData['Canonical SMILES'].to_list()]    
    data['molecule'] = [RDKitCarefulSanitize(i)[0] for i in data['molecule'] ]
    data['molecule'] = StandardiseMolecules(data['molecule'])
    data['denticity'] = list(InpData['Theoretical denticity'])
    data['database ID'] = ['SDI'+'0'*(6-len(str(i)))+str(i) for i in InpData['Siderophore ID']]
    data['smiles'] = [Chem.MolToSmiles(i) for i in data['molecule']]
    data['name'] = InpData['Siderophore name']
    data['type'] = InpData['Ligand Type']
    data['metal'] = ['Fe']*len(data['smiles'])
    return data

def CheckDatasets(DS): 
    B =  [bool(m and s) for m,s in zip(DS['molecule'],DS['smiles'])]
    ODS = {k:[i for i,b in zip(v,B) if b] for k,v in DS.items()}
    return ODS
def Calculate_Overlap(ChelDatasets,NPData,stereo=True):
    def removeStereo(s):
        m = Chem.MolFromSmiles(s,sanitize=False)
        m,e = RDKitCarefulSanitize(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)
    if not stereo:
        print('\tRemoving stereochemistry')
        CDs = [[removeStereo(i) for i in d['smiles']] for d in ChelDatasets]
        NPD = [removeStereo(i) for i in NPData['smiles']]
    else:
        CDs = [d['smiles'] for d in ChelDatasets]
        NPD = NPData['smiles']
    OutInfo = [[';'.join([k for j,k in zip(NPD,NPData['NP ID']) if i==j]) for i in tqdm(d)] for d in CDs]
    return OutInfo

def GetDescriptors(m):
    try:
        ## list of descriptors used in the study to define relevant chemical space
        descs = {
            'MolWt'                 : Descriptors.MolWt(m),
            'molecular complexity'  : DescFn.bottcher_complexity(m),
            'BertzCT'               : Descriptors.BertzCT(m), 
            'MolLogP'               : Descriptors.MolLogP(m), 
            'apol'                  : DescFn.CalculateAPol(m),
            'bpol'                  : DescFn.CalculateBPol(m),
            'proportion O'          : DescFn.prop_O(m),
            'proportion N'          : DescFn.prop_N(m),
            'proportion S'          : DescFn.prop_S(m),
            'LabuteASA'             : Chem.rdMolDescriptors.CalcLabuteASA(m),
            'RotRatio'              : DescFn.RotRatio(m),
            'Molecular Refractivity': Descriptors.MolMR(m),
            'proportion aromatic'   : DescFn.get_prop_aromatic(m),
            'proportion ring'       : DescFn.get_prop_ring(m),
            'TPSA'                  : Descriptors.TPSA(m),
            'NumHAcceptors'         : Descriptors.NumHAcceptors(m),
            'NumHDonors'            : Descriptors.NumHDonors(m)
        }
    except Exception as e:
        descs = ''
        # print(e)
    return descs

def standardscalar(X):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

## NEEDS TO BE OUTSIDE OF __MAIN__ to be used in multiprocessing
pse = Chem.GetPeriodicTable()
alkali_metals =         [3,11,19,37,55,87]
alkaline_earth_metals = [4,12,20,38,56,88]
post_transition_metals = [13,31,49,50,81,82,83,113,114,115,116]
lanthanides = [i for i in list(range(57,72,1))]
actinides = [i for i in list(range(89,104,1))]
transition_metals = list(range(21,31))+list(range(39,49))+list(range(72,81))+list(range(104,113))
TransitionMetalSymbols = [pse.GetElementSymbol(i) for i in transition_metals]
TransitionMetalSymbolsUpper = [i.upper() for i in TransitionMetalSymbols]

# input data paths
ccdc_data_path = 'input_data/Metalloid+nonMet-TM-3D-NoDis.gcd'
SDR_path = 'input_data/Siderite_2025_paper_tableS2.csv'
pdb_ligs_path = 'input_data/Components-smiles-stereo-oe.smi'
l_path = 'input_data/natural_product_dbs/LOTUS_DB.smi'
c_path = 'input_data/natural_product_dbs/coconut_csv-05-2025.csv'
s_path = 'input_data/natural_product_dbs/supernatural_3.csv'

PROCS = 12
## Ligands that bind via these elements are not considered to support comparison to biological chelators, used in the PDB ligands
ExcludedSymbolsUpper = [i.upper() for i in [pse.GetElementSymbol(i) for i in [5,6,14,32,33,34,51,52]]]
## Loading PDB ligands from file
PDBLigInfo = {}
with open(pdb_ligs_path,'r') as f:
    for l in f:
        smi, code, name = l.rstrip().split('\t')
        if not '.' in smi:
            PDBLigInfo[code] = [smi,name]
## some manual fixes, adding in a fixed version of the 8L5 ligand
PDBLigInfo['8L5'] = ['CC(C)C[C@@H](C(=O)NCN)NP(=O)(CNC(=O)OCc1ccccc1)O','~{n}-[(2~{s})-1-(aminomethylamino)-4-methyl-1-oxidanylidene-pentan-2-yl]-(phenylmethoxycarbonylaminomethyl)phosphonamidic acid;hydrogen']

if __name__ == "__main__":
    print('Generating CSD ligands')
    if not os.path.isfile('output_data/CSD_Ligs.pk1'):
        CSD = Generate_CSD_data(ccdc_data_path)
        with open('output_data/CSD_Ligs.pk1','wb') as handle:   
            pickle.dump(CSD, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Generating PDB ligands')
    if not os.path.isfile('output_data/PDBS_Ligs.pk1'):
        PDBS = Generate_PDB_ligands()
        with open('output_data/PDBS_Ligs.pk1','wb') as handle:   
                pickle.dump(PDBS, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Generating NP data')
    if not os.path.isfile('output_data/NP_Ligs.pk1'):
        NP = Generate_NP_data(COCONUTpath=c_path,LOTUSpath=l_path,SUPERNATURALpath=s_path)
        with open('output_data/NP_Ligs.pk1','wb') as handle:   
                pickle.dump(NP, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Generating Sideite data')
    if not os.path.isfile('output_data/SDR_Ligs.pk1'):
        print('\tGenerating Siderite data')
        SDR = Generate_Siderite_data(SDR_path)
        with open('output_data/SDR_Ligs.pk1','wb') as handle:   
                pickle.dump(SDR, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Generating descriptors')
    if not os.path.isfile('output_data/Dataset-descs.pk1'):
        print('\tLoading data')
        with open('output_data/CSD_Ligs.pk1','rb') as handle:
            CSD = pickle.load(handle)
        with open('output_data/PDBS_Ligs.pk1','rb') as handle:
            PDBS = pickle.load(handle)
        with open('output_data/SDR_Ligs.pk1','rb') as handle:
            SDR = pickle.load(handle)
        print('\tChecking data')
        CSD = CheckDatasets(CSD)
        PDBS = CheckDatasets(PDBS)
        SDR = CheckDatasets(SDR)

        print('\tGenerating Descriptors')
        CSD['descriptors'] = [GetDescriptors(m) for m in tqdm(CSD['molecule'],desc='CSD',total=len(CSD['molecule']))]
        PDBS['descriptors'] = [GetDescriptors(m) for m in tqdm(PDBS['molecule'],desc='PDB',total=len(PDBS['molecule']))]
        SDR['descriptors'] = [GetDescriptors(m) for m in tqdm(SDR['molecule'],desc='Siderite',total=len(SDR['molecule']))]
        with open('output_data/Dataset-descs.pk1','wb') as handle:  
            pickle.dump((CSD,SDR,PDBS), handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open('output_data/Dataset-descs.pk1','rb') as handle:   
        CSD,SDR,PDBS = pickle.load(handle)  

    print('Calculating CSD-NP and PDB-NP Overlap')
    if not os.path.isfile('output_data/Dataset-NP-overlap.pk1'):
        print('\tLoading NP dataset (long)')
        with open('output_data/NP_Ligs.pk1','rb') as handle:
            NP = pickle.load(handle)
        NP = CheckDatasets(NP)
        print('calculating overlap')
        CSDNPInfo,PDBNPInfo = Calculate_Overlap((CSD,PDBS),NP,stereo=False)
        with open('output_data/Dataset-NP-overlap.pk1','wb') as handle:  
            pickle.dump((CSDNPInfo,PDBNPInfo), handle, protocol=pickle.HIGHEST_PROTOCOL) 
        del NP
    with open('output_data/Dataset-NP-overlap.pk1','rb') as handle:   
        CSDNPInfo,PDBNPInfo = pickle.load(handle)  

    print('Saving final dataset')
    CSD['database'] = ['Cambridge Structural Database']*len(CSD['smiles'])
    PDBS['database'] = ['Protein Data Bank']*len(PDBS['smiles'])
    SDR['database'] = ['Siderite']*len(SDR['smiles'])
    CSD['type'] = ['']*len(CSD['smiles'])
    CSD['name'] = ['']*len(CSD['smiles'])
    PDBS['type'] = ['']*len(PDBS['smiles'])
    PDBS['name'] = ['']*len(PDBS['smiles'])
    CSD['NP info'] = CSDNPInfo
    PDBS['NP info'] = PDBNPInfo
    SDR['NP info'] = SDR['database ID']
    TotalKeys = ['smiles','database','database ID','NP info','denticity','metal','type','name']
    datasets = {k:[i for d in [CSD,PDBS,SDR] for i in d[k]] for k in TotalKeys+['descriptors']}
    ## removing duplicates
    datasetBool = [True]*len(datasets['smiles'])
    for n,smi1 in enumerate(datasets['smiles']):
        if datasetBool[n]:
            for m,smi2 in enumerate(datasets['smiles'][n+1:]):
                if datasetBool[m] and smi1==smi2:
                    datasetBool[m] = False
                    datasets['database'][n] = ';'.join(set(datasets['database'][n].split(';')+datasets['database'][m].split(';')))
                    KeepInfo = [not i in datasets['database ID'][m].split(';') for i in datasets['database ID'][m].split(';')]
                    for d in ['database ID','NP info','denticity','metal']:
                        datasets[d][n] = ';'.join(datasets[d][n].split(';')+[i for i,j in zip(datasets[d][m].split(),KeepInfo) if j])
    datasets = {k:[v for v,b in zip(datasets[k],datasetBool) if b ] for k in datasets}
    datasetBool = [all([datasets[k][i] for k in datasets.keys() if not k in ['NP info','type','name']]) for i in range(len(datasets['smiles']))]
    print(len(datasetBool)-sum(datasetBool),'molecules removed due to missing information')
    datasets = {k:[v for v,b in zip(datasets[k],datasetBool) if b ] for k in datasets}
    wanted_descs = ['MolWt'
                    ,'LabuteASA'
                    ,'molecular complexity'
                    ,'BertzCT'
                    ,'apol',
                    'bpol']+[
                            'NumHAcceptors'
                            ,'NumHDonors'
                            ,'TPSA'
                            ,'MolLogP'
                            ]+['proportion O','proportion N','proportion S',]+['RotRatio','proportion ring','proportion aromatic']
    print('\tcalculating PCA')
    ScaleData = standardscalar([[v for k,v in i.items() if k in wanted_descs] for i in datasets['descriptors']])
    pca = PCA(n_components=2)
    pca.fit(ScaleData)
    PCAcoeff = pca.components_.T
    datasets['PC1'],datasets['PC2'] = zip(*pca.transform(ScaleData))
    print('\tSaving final dataset')
    with open('output_data/TotalDataset.pk1','wb') as handle:   
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('\tWritting final dataset CSV')
    with open('output_data/Feeney_Master_Chelator_Database_2026.csv','w', encoding="utf-8") as f:  
        datasets = list(zip(*[datasets[k] for k in TotalKeys+['PC1','PC2','descriptors']]))
        f.write(','.join(TotalKeys+['PC1','PC2']+wanted_descs))
        for row in datasets:
            f.write('\n'+','.join([str(i) for i in list(row[:-1])+list([row[-1][k] for k in wanted_descs])]))