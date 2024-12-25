from rdkit import Chem
import copy
import sys
sys.path.append(r"/Users/valentinnovikov/OneDrive/0_Spain/3_Data Science/3_NMR/OdanChem_ETL_draft") #импортировать соседнюю репу: https://github.com/Oleafan/OdanChem_ETL_draft/tree/oleg_dev (ветка oleg_dev, а не main)
import predict_carbon_full as pc #Korean mpnn for CNMR prediction

import numpy as np

def _cut_out_fragment (mol, atom_list):
    #вырезает фрагмент, состоящий из перечисленных в atom_list атомов
    mol = Chem.AddHs(mol) 
    Chem.Kekulize(mol,clearAromaticFlags=True) 
    edmol = Chem.EditableMol(mol) 
    atoms_no_implicit = []
    
    for atom_idx in atom_list:
        atom = mol.GetAtomWithIdx(atom_idx)
        for atom_low in atom.GetNeighbors():
            atom_low_idx = atom_low.GetIdx()
            if atom_low_idx not in atom_list and atom_low.GetSymbol() != 'H':
                edmol.RemoveBond(atom_idx,atom_low_idx)
                atoms_no_implicit.append(atom_low_idx)
                atoms_no_implicit.append(atom_idx)
                
    mol_new = edmol.GetMol()             
    for atom_idx in atoms_no_implicit:
        mol_new.GetAtomWithIdx(atom_idx).SetNoImplicit(True)
    
    Chem.SanitizeMol(mol_new)
    Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
    return Chem.RemoveHs(mol_new)

def get_fp_frags(mol, radius):
    """функция пилит молекулы на фрагменты радиусом radius"""
    frag_smiles_list = []
    for atom in mol.GetAtoms(): #идем по всем атомам
        
        atom_idx = atom.GetIdx() #индекс центрального атома в фрагменте
        generation_idx = 0 
        atom_list = [[atom_idx]] #список вида [[atom_idx], [list of 1st gen atoms], [list of second gen atoms], ...]
        while generation_idx < radius:
            idxes = atom_list[-1]
            temp_idx_list = []
            for idx in idxes:
                atom_ = mol.GetAtomWithIdx(idx)
                for atom_low in atom_.GetNeighbors():
                    temp_idx_list.append(atom_low.GetIdx())
            atom_list.append(temp_idx_list)
            generation_idx += 1
        
        flat_atom_list = []
        for idx_list in atom_list:
            flat_atom_list += idx_list
        atom_no_set = set(flat_atom_list)
        
        mol_new = _cut_out_fragment (mol, atom_no_set)
        
        target_mol_frag = None
        for idx_list, mol_frag in zip(Chem.GetMolFrags(mol_new), Chem.GetMolFrags(mol_new, asMols=True)):
            if len(atom_no_set - set(idx_list)) == 0:
                target_mol_frag  = mol_frag
                break
                
        if target_mol_frag is not None:
            frag_smiles_list.append(Chem.MolToSmiles(target_mol_frag ))
            
    return list(set(frag_smiles_list))

def _cut_single_frag (mol, atom_list):
    
    mol_new = _cut_out_fragment(mol, atom_list )
    target_mol_frag = None
    for idx_list, mol_frag in zip(Chem.GetMolFrags(mol_new), Chem.GetMolFrags(mol_new, asMols=True)):
        if len(set(atom_list) - set(idx_list)) == 0:
            target_mol_frag  = mol_frag
            break
    return target_mol_frag

def _is_included(patt, mol):
    smarts = Chem.MolToSmarts(patt)
    return mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))

def _get_real_valence(atom_idx, mol):
    mol = copy.deepcopy(mol)
    Chem.Kekulize(mol)
    atom = mol.GetAtomWithIdx(atom_idx)
    
    radicals = atom.GetNumRadicalElectrons()
    
    for neigh in atom.GetNeighbors():
        bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(),neigh.GetIdx()).GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            radicals += 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            radicals += 2
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            radicals += 3
    return radicals

def merge_fragments(frag_1, frag_2):
    #функция объединяет два фрагмента по пересекающейся общей части
    if frag_1.GetNumAtoms() >= frag_2.GetNumAtoms():
        frag_1_ = copy.deepcopy(frag_1)
        frag_2_ = copy.deepcopy(frag_2)
    else: 
        frag_1_ = copy.deepcopy(frag_2)
        frag_2_ = copy.deepcopy(frag_1)
    
    init_num_atoms = frag_1_.GetNumAtoms() #в конце берем только те фрагменты, в котороых от этого количества атомов.
    #frag_2_ - более мелкая молекула
    #найти атом, который со всеми соседями входит во вторую молекулу
    Chem.Kekulize(frag_2_)
    Chem.Kekulize(frag_1_)
    list_of_idx_patt = []
    for atom in frag_2_.GetAtoms(): #по атомам более мелкой молекулы
        idx_list = [atom.GetIdx()] + [neigh.GetIdx() for neigh in atom.GetNeighbors()]
        if len(idx_list) > 2: #есть хотя бы 3 атома
            patt = _cut_single_frag(frag_2_, idx_list) 
            if _is_included(patt, frag_1_): #если эти атомы входят в целевую молекулу
                for idx in idx_list: 
                    _atom = frag_2_.GetAtomWithIdx(idx) 
                    for neigh in _atom.GetNeighbors():
                        if neigh.GetIdx() not in idx_list:
                            temp_idx_list = idx_list + [neigh.GetIdx() ]
                            patt = _cut_single_frag(frag_2_, temp_idx_list)
                            if _is_included(patt, frag_1_): #если данный сосед в сочетании с остальным остовом входит в целевую молекулу
                                idx_list.append(neigh.GetIdx() )
                appending = True
                for item in list_of_idx_patt:
                    if set(idx_list) - set(item) == set():
                        appending = False
                if appending:    
                    list_of_idx_patt.append(idx_list)

    matches = []
    for idx_list in list_of_idx_patt:
        patt = _cut_single_frag(frag_2_, idx_list)
        for atom in patt.GetAtoms():
            atom.SetNumRadicalElectrons(0)
        atoms_1_lst = frag_1_.GetSubstructMatches(patt)
        atoms_2_lst = frag_2_.GetSubstructMatches(patt)

        for atoms_1 in atoms_1_lst:
            for atoms_2 in atoms_2_lst:
                match_list = [(x,y) for x,y in zip(atoms_1, atoms_2) ]
                true_match = True

                for pair in match_list:
                    atom_1 = frag_1_.GetAtomWithIdx(pair[0])
                    atom_2 = frag_2_.GetAtomWithIdx(pair[1]) 

                    if _get_real_valence(pair[0], frag_1_) != _get_real_valence(pair[1], frag_2_):
                        true_match = False        
                if true_match:
                    matches.append(match_list)    
    if len(matches) == 0:
        return None
    
    result = []
    for working_match in matches:

        #frag_2_ - более мелкая молекула
        #база для наращивания новых фрагментов - frag_1_
        combo = Chem.CombineMols(frag_1_,frag_2_)
        edcombo = Chem.EditableMol(combo)
        atom_no_shift = frag_1_.GetNumAtoms()
        match1 = [x[0] for x in working_match] #атомы во frag_1_, которые вошли в матч
        match2 = [x[1] for x in working_match] #атомы во frag_2_, которые вошли в матч

        for pair in working_match:
            atom_1 = frag_1_.GetAtomWithIdx(pair[0])
            atom_2 = frag_2_.GetAtomWithIdx(pair[1]) 
            if atom_1.GetNumRadicalElectrons() > atom_2.GetNumRadicalElectrons():
                #значит в frag_2_ есть что-то что надо законнектить с frag_1
                for neigh in atom_2.GetNeighbors():
                    if neigh.GetIdx() not in match2:
                        #если сосед не в матче - получаем тип связи соседа с atom_2 
                        bond_type = frag_2_.GetBondBetweenAtoms(pair[1],neigh.GetIdx()).GetBondType()
                        #удаляем эту связь
                        edcombo.RemoveBond(pair[1] + atom_no_shift,neigh.GetIdx() + atom_no_shift)
                        #добавляем новую связь с frag_1

                        edcombo.AddBond(pair[0],neigh.GetIdx()+ atom_no_shift,order=bond_type)

                        combo = edcombo.GetMol()
                        combo.GetAtomWithIdx(pair[0]).SetNumRadicalElectrons(atom_2.GetNumRadicalElectrons())
                        edcombo = Chem.EditableMol(combo)


        combo = edcombo.GetMol()
        frags = Chem.GetMolFrags(combo, asMols=True, sanitizeFrags = False) #
        
        for frag in frags:
            try:
                Chem.SanitizeMol(frag)
                if frag.GetNumAtoms() > init_num_atoms:
                    result.append(frag)
            except Exception as e:
                pass
        
    moldict = {}
    for frag in result:
        try:
            moldict[frag] = frag.GetNumAtoms()
        except:
            pass
    if len(moldict) == 0:
        return None
    moldict = sorted(moldict.items(), key=lambda item: item[1], reverse=True)    
    return result #[moldict[0][0]]  -если так, то возвращает самую большую по числу атомов конструкцию. Можно возвращать вообще все, но тогда будет слишком дохрена результатов


def cyclize_rings(mol_frag, max_ring_size=8, check_aromatic = True):
    
    bond_type_dict = {1: Chem.rdchem.BondType.SINGLE, 
                      2: Chem.rdchem.BondType.DOUBLE,
                      3: Chem.rdchem.BondType.DOUBLE,
                      4: Chem.rdchem.BondType.DOUBLE}
    
    #если в молекуле есть несколько разных радикальных хвостов, пытается их "жадно" замкнуть друг на друга
    #если при этом получается больше циклов размером от 6 - круто, возвращает результат. Связь делает максимум двойную 
    
    radical_atoms_dict = {} #{atom:num_rad_els}
    for atom in mol_frag.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            radical_atoms_dict[atom.GetIdx()] = atom.GetNumRadicalElectrons()
    radical_atoms_dict = dict(sorted(list(radical_atoms_dict.items()), key = lambda x: x[1])) #сортируем по увеличению числа радикалов
    
    #пары генерируем только от менее радикальных атомов к более (то есть по возрастанию radical_atoms_dict)
    radical_atoms_list = list(radical_atoms_dict.keys())
    pair_list = []
    for atom_idx_1, _ in enumerate(radical_atoms_list):
        for atom_idx_2, _ in enumerate(radical_atoms_list):
            if atom_idx_2 > atom_idx_1:
                pair_list.append([radical_atoms_list[atom_idx_1], radical_atoms_list[atom_idx_2] ])
    
    ringsize_list = list(range(6,max_ring_size+1) )
    result_mols = [mol_frag] #и дополняем зацикленными фрагментами
    
    for pair in pair_list:
        #Нет ли уже связи между этими атомами
        if pair[1] in [atom.GetIdx() for atom in  mol_frag.GetAtomWithIdx(pair[0]).GetNeighbors()]:
            continue
        
        rings_state_init = {}
        for atom_idx in pair:
            rings_state_init[atom_idx] = [size for size in ringsize_list if mol_frag.GetAtomWithIdx(atom_idx).IsInRingSize(size)]
            
        arom_state_init = {}
        for atom_idx in pair:
            arom_state_init[atom_idx] = mol_frag.GetAtomWithIdx(atom_idx).GetIsAromatic()
        
        edmol = Chem.EditableMol(mol_frag)
        edmol.AddBond(pair[0],pair[1],order=bond_type_dict[radical_atoms_dict[pair[0]]])
        combo = edmol.GetMol()
        combo.GetAtomWithIdx(pair[1]).SetNumRadicalElectrons(radical_atoms_dict[pair[1]] - radical_atoms_dict[pair[0]])
        combo.GetAtomWithIdx(pair[0]).SetNumRadicalElectrons(0)
        
        Chem.SanitizeMol(combo)
        
        rings_state_final = {}
        for atom_idx in pair:
            rings_state_final[atom_idx] = [size for size in ringsize_list if combo.GetAtomWithIdx(atom_idx).IsInRingSize(size)]
            
        arom_state_final = {}
        for atom_idx in pair:
            arom_state_final[atom_idx] = combo.GetAtomWithIdx(atom_idx).GetIsAromatic()
        
        appending = False
        for atom_idx in pair:
            if len(rings_state_final[atom_idx] ) > len(rings_state_init[atom_idx] ): #если хоть один из атомов теперь вовлечен в какой-то новый цикл размером >= 6
                if check_aromatic:
                    if arom_state_final[atom_idx] and not arom_state_init[atom_idx]: #если при этом возникла ароматичность
                        appending = True 
                else:
                    appending = True 
        
        if appending: 
            result_mols.append(combo)
    return result_mols


def close_readicals(mol):
    #function close all radical atoms with hydrogens
    closing_atom = Chem.MolFromSmiles('[H]') 
    
    closed_mol = copy.deepcopy(mol)
    radical_atom_dict = {}
    for atom in closed_mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            radical_atom_dict[atom.GetIdx()] = atom.GetNumRadicalElectrons()
    
    atom_no_shift = 0
    basic_atom_number = closed_mol.GetNumAtoms()
    
    for atom_idx in radical_atom_dict:
        for _ in range(radical_atom_dict[atom_idx]):
            combo = Chem.CombineMols(closed_mol,closing_atom)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(atom_idx, basic_atom_number + atom_no_shift, order=Chem.rdchem.BondType.SINGLE)
            
            closed_mol = edcombo.GetMol()
            closed_mol.GetAtomWithIdx(atom_idx).SetNumRadicalElectrons(
                closed_mol.GetAtomWithIdx(atom_idx).GetNumRadicalElectrons()-1)
            closed_mol.GetAtomWithIdx(basic_atom_number + atom_no_shift).SetNumRadicalElectrons(0)
            
            atom_no_shift += 1
    return closed_mol


def get_num_carbon_atoms(mol):
    num = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            num+= 1
    return num




def get_predicted_ppm (mol, averag_num = 2):
    #функция возвращает список предсказанных ppm. По умолчанию делает это 2 раза и усредняет 
    #полученные предсказания
    try:
        
        if averag_num == 1:
            ppm_dict_list = pc.get_symm_ppm_dict(mol, seed = 0)
            predicted_ppm = np.array([x['ppm'] for x in ppm_dict_list])
        else: 
            for idx in range(averag_num):
                if idx == 0:
                    ppm_dict_list = pc.get_symm_ppm_dict(mol, seed = idx)
                    predicted_ppm = np.array([x['ppm'] for x in ppm_dict_list])
                else:
                    ppm_dict_list = pc.get_symm_ppm_dict(mol, seed = idx)
                    predicted_ppm_temp = np.array([x['ppm'] for x in ppm_dict_list])
                    
                    predicted_ppm = np.vstack(([predicted_ppm,predicted_ppm_temp]))
            
            predicted_ppm = np.mean(predicted_ppm, axis = 0)
            
        return predicted_ppm
    except: 
        return None
    
    
#from spec_check_functions
def _find_min (ppm_array, ppm_ind):
    return np.argmin(np.abs(ppm_array - ppm_ind))

def _find_corr_index(long_ppm_array, short_ppm_array):
    long_ppm_array.sort()
    long_ppm_local = long_ppm_array.copy()
    short_ppm_array.sort()
    short_ppm_local = short_ppm_array.copy()
    
    long_idx_list = []
    
    for ppm in short_ppm_local:
        idx = _find_min(long_ppm_local, ppm)
        long_idx_list.append(idx)
        long_ppm_local[idx] += 10000
        
    return long_idx_list

def check_mae_realppm (real_ppm, predicted_ppm, return_real_list = True):
    
    real_ppm = np.array(real_ppm)
    predicted_ppm = np.array(predicted_ppm)
    
    real_ppm.sort()
    predicted_ppm.sort()
    
    if len(real_ppm) == len(predicted_ppm):
        mae = np.mean(np.abs(real_ppm-predicted_ppm))
        maxdif = np.max(np.abs(real_ppm-predicted_ppm))
        
        if return_real_list:
            return mae, maxdif, real_ppm
        else:
            return mae, maxdif
    
    elif len(real_ppm) < len(predicted_ppm):
        predicted_idx_list = _find_corr_index(predicted_ppm, real_ppm)
        predicted_ppm_croped = predicted_ppm[predicted_idx_list]
        predicted_ppm_croped.sort()
        mae = np.mean(np.abs(real_ppm-predicted_ppm_croped))
        maxdif = np.max(np.abs(real_ppm-predicted_ppm_croped))
        
        if return_real_list:
            return mae, maxdif, real_ppm
        else:
            return mae, maxdif
        
    elif len(real_ppm) > len(predicted_ppm):
        real_idx_list = _find_corr_index(real_ppm,predicted_ppm)
        real_ppm_croped = real_ppm[real_idx_list]
        real_ppm_croped.sort()
        mae = np.mean(np.abs(real_ppm_croped-predicted_ppm))
        maxdif = np.max(np.abs(real_ppm_croped-predicted_ppm))

        if return_real_list:
            return mae, maxdif, real_ppm_croped
        else:
            return mae, maxdif