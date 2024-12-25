import numpy as np
np.random.seed(0)
import os, sys
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(0)

import dgl
from dgl.convert import graph
from dgl.nn.pytorch import NNConv, Set2Set

from sklearn.metrics import mean_absolute_error

import time

#based on https://github.com/seokhokang/nmr_mpnn_pytorch 

class GraphDataset():

    def __init__(self, name='nmrshiftdb2'):

        self.name = name
        self.load()


    def load(self):
        [mol_dict] = np.load('./data/dataset_graph.npz', allow_pickle=True)['data']

        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
                
        self.shift = mol_dict['shift']
        self.mask = mol_dict['mask']
        self.smi = mol_dict['smi']

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
        

    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx:idx+1].astype(int)
        shift = self.shift[self.n_csum[idx]:self.n_csum[idx+1]].astype(float)
        mask = self.mask[self.n_csum[idx]:self.n_csum[idx+1]]
        
        return g, n_node, shift, mask
        
        
    def __len__(self):
        return self.n_node.shape[0]
    
def collate_reaction_graphs(batch):

    gs, n_nodes, shifts, masks = map(list, zip(*batch))
    
    gs = dgl.batch(gs)

    n_nodes = torch.LongTensor(np.hstack(n_nodes))
    shifts = torch.FloatTensor(np.hstack(shifts))
    masks = torch.BoolTensor(np.hstack(masks))
    
    return gs, n_nodes, shifts, masks


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass

class nmrMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 node_feats = 64, embed_feats = 256,
                 num_step_message_passing = 5,
                 num_step_set2set = 3, num_layer_set2set = 1,
                 hidden_feats = 512, prob_dropout = 0.1):
        
        super(nmrMPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, node_feats), nn.Tanh()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, node_feats * node_feats), nn.ReLU()
        )
        
        self.gnn_layer = NNConv(
            in_feats = node_feats,
            out_feats = node_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.gru = nn.GRU(node_feats, node_feats)
        
        self.readout = Set2Set(input_dim = node_feats * (1 + num_step_message_passing),
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)
                               
        self.predict = nn.Sequential(
            nn.Linear(node_feats * (1 + num_step_message_passing) * 3, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, 1)
        )                           
                               
    def forward(self, g, n_nodes, masks):
        
        def embed(g):
            
            node_feats = g.ndata['node_attr']
            node_feats = self.project_node_feats(node_feats)

            edge_feats = g.edata['edge_attr']

            node_aggr = [node_feats]
            for _ in range(self.num_step_message_passing):
                msg = self.gnn_layer(g, node_feats, edge_feats).unsqueeze(0)
                _, node_feats = self.gru(msg, node_feats.unsqueeze(0))
                node_feats = node_feats.squeeze(0)
                
                node_aggr.append(node_feats)
                
            node_aggr = torch.cat(node_aggr, 1)
            
            return node_aggr

        node_embed_feats = embed(g)
        graph_embed_feats = self.readout(g, node_embed_feats)        
        graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)

        out = self.predict(torch.hstack([node_embed_feats, graph_embed_feats])[masks]).flatten()

        return out

        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, n_forward_pass = 5, cuda = torch.device('cpu')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-10)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6, verbose=True)

    max_epochs = 500
    val_y = np.hstack([inst[-2][inst[-1]] for inst in iter(val_loader.dataset)])
    val_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs, n_nodes, shifts, masks = batchdata
            
            shifts = (shifts[masks] - train_y_mean) / train_y_std
            
            inputs = inputs.to(cuda)
            n_nodes = n_nodes.to(cuda)
            shifts = shifts.to(cuda)
            masks = masks.to(cuda)
            
            predictions = net(inputs, n_nodes, masks)
            
            loss = torch.abs(predictions - shifts).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item() * train_y_std

        #print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  train_size, train_size, train_loss, (time.time()-start_time)/60))
    
        # validation
        val_y_pred = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
        val_loss = mean_absolute_error(val_y, val_y_pred)
        
        val_log[epoch] = val_loss
        if epoch % 10 == 0: print('--- validation epoch %d, processed %d, current MAE %.3f, best MAE %.3f, time elapsed(min) %.2f' %(epoch, val_loader.dataset.__len__(), val_loss, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
        
        lr_scheduler.step(val_loss)
        
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            break

    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cpu')):

    net.eval()
    MC_dropout(net)
    tsty_pred = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs = batchdata[0].to(cuda)
            n_nodes = batchdata[1].to(cuda)
            masks = batchdata[3].to(cuda)

            tsty_pred.append(np.array([net(inputs, n_nodes, masks).cpu().numpy() for _ in range(n_forward_pass)]).transpose())

    tsty_pred = np.vstack(tsty_pred) * train_y_std + train_y_mean
    
    return np.mean(tsty_pred, 1)


def add_mol(mol_dict, mol):

    def _DA(mol):

        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

    def _chirality(atom):

        if atom.HasProp('Chirality'):
            #assert atom.GetProp('Chirality') in ['Tet_CW', 'Tet_CCW']
            c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
        else:
            c_list = [0, 0]

        return c_list

    def _stereochemistry(bond):

        if bond.HasProp('Stereochemistry'):
            #assert bond.GetProp('Stereochemistry') in ['Bond_Cis', 'Bond_Trans']
            s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
        else:
            s_list = [0, 0]

        return s_list    
        

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    D_list, A_list = _DA(mol)
    rings = mol.GetRingInfo().AtomRings()
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    
    node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    
    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
        bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]   
        
        edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)
    
    return mol_dict


atom_list = ['Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi']
charge_list = [1, 2, 3, -1, -2, -3, 0]
degree_list = [1, 2, 3, 4, 5, 6, 0]
valence_list = [1, 2, 3, 4, 5, 6, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]

bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))


model_path = 'nmr_model.pt'
node_dim = 69
edge_dim = 8
net = nmrMPNN(node_dim, edge_dim) 
net.load_state_dict(torch.load(model_path))

mol_dict = {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': [],
            'shift': [],
            'mask': [],
            'smi': []}

class GraphDataset():

    def __init__(self, mol_dict, name='exp_test'):

        self.name = name
        self.load(mol_dict)


    def load(self, mol_dict):
        #mol_dict = np.load('./data/dataset_graph.npz', allow_pickle=True)['data']

        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
                
        self.shift = mol_dict['shift']
        self.mask = mol_dict['mask']
        self.smi = mol_dict['smi']

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx:idx+1].astype(int)
        shift = self.shift[self.n_csum[idx]:self.n_csum[idx+1]].astype(float)
        mask = self.mask[self.n_csum[idx]:self.n_csum[idx+1]]
        
        return g, n_node, shift, mask
        
        
    def __len__(self):

        return self.n_node.shape[0]



    
def get_nmr_shiftspreloaded(mol, seed = 0, net = net):
        
    atom_list = ['Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi']
    charge_list = [1, 2, 3, -1, -2, -3, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]

    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

    rdBase.DisableLog('rdApp.error') 
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

    mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'shift': [],
                'mask': [],
                'smi': []}
    
    try:
        Chem.SanitizeMol(mol)
        si = Chem.FindPotentialStereo(mol)
        for element in si:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        assert '.' not in Chem.MolToSmiles(mol)
    except:
        pass

    for j, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == 'C':
            atom.SetDoubleProp('shift', 0)
            atom.SetBoolProp('mask', 1)  
        else:
            atom.SetDoubleProp('shift', 0)
            atom.SetBoolProp('mask', 0)

    mol = Chem.RemoveHs(mol)
    mol_dict = add_mol(mol_dict, mol)

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    mol_dict['shift'] = np.hstack(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])
    data = GraphDataset(mol_dict)
    
    torch.manual_seed(seed)
    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=False, collate_fn=collate_reaction_graphs, drop_last=True)
    
    

    for batchdata in data_loader:
        inputs, n_nodes, shifts, masks = batchdata
        break
        
    predictions = net(inputs, n_nodes, masks)
    
    train_y_mean = 95.86915330336912
    train_y_std  = 51.61745076037435
    
    ppms = predictions*train_y_std+train_y_mean
    
    return ppms.detach().numpy(), masks.detach().numpy()

def get_symm_ppm_dict(mol, seed = 0, net = net):
    ppms, mask = get_nmr_shiftspreloaded(mol, seed = seed, net = net) 
    
    symm_list = np.array(list(Chem.CanonicalRankAtoms(mol, breakTies=False)))
    
    atom_idx_list =  np.array([atoms.GetIdx() for atoms in mol.GetAtoms()])
    atom_sym_list = np.array([atoms.GetSymbol() for atoms in mol.GetAtoms()])
    
    carbons = atom_sym_list == 'C'
    atom_idx_list = atom_idx_list[carbons]
    symm_list = symm_list[carbons]
    
    mol_dict = {'atom_no':atom_idx_list, 'Symmetry_num':  symm_list, 'ppm': ppms}   
    symm_list = np.array(symm_list)
    atom_idx_list = np.array(atom_idx_list)
    
    ppm_symm_list = []
    for symm_no in list(set(symm_list)):
        symm_ppm_dict = {}
        curr_symm = symm_list == symm_no
        symm_ppm_dict.update({'atom_no': atom_idx_list[curr_symm]})
        symm_ppm_dict.update({'ppm': np.round(np.mean(ppms[curr_symm]), 2)})
        ppm_symm_list.append(symm_ppm_dict)
        
    return ppm_symm_list

def get_predicted_ppm (inch, averag_num = 3):
    #функция возвращает список предсказанных ppm. По умолчанию делает это три раза и усредняет полученные предсказания
    try:
        try:
            mol = Chem.MolFromInchi(inch)
        except: 
            mol = Chem.MolFromSmiles(inch)
        
        if averag_num == 1:
            ppm_dict_list = get_symm_ppm_dict(mol, seed = 0)
            predicted_ppm = np.array([x['ppm'] for x in ppm_dict_list])
        else: 
            for idx in range(averag_num):
                if idx == 0:
                    ppm_dict_list = get_symm_ppm_dict(mol, seed = idx)
                    predicted_ppm = np.array([x['ppm'] for x in ppm_dict_list])
                else:
                    ppm_dict_list = get_symm_ppm_dict(mol, seed = idx)
                    predicted_ppm_temp = np.array([x['ppm'] for x in ppm_dict_list])
                    
                    predicted_ppm = np.vstack(([predicted_ppm,predicted_ppm_temp]))
            
            predicted_ppm = np.mean(predicted_ppm, axis = 0)
            
        return predicted_ppm
    except: 
        pass
    
def get_predicted_ppm_from_molfile (inp_molfile, averag_num = 3):
    #функция возвращает список предсказанных ppm. По умолчанию делает это три раза и усредняет полученные предсказания
    try:
        mol = Chem.MolFromMolBlock(inp_molfile)
        
        if averag_num == 1:
            ppm_dict_list = get_symm_ppm_dict(mol, seed = 0)
            predicted_ppm = np.array([x['ppm'] for x in ppm_dict_list])
        else: 
            for idx in range(averag_num):
                if idx == 0:
                    ppm_dict_list = get_symm_ppm_dict(mol, seed = idx)
                    predicted_ppm = np.array([x['ppm'] for x in ppm_dict_list])
                else:
                    ppm_dict_list = get_symm_ppm_dict(mol, seed = idx)
                    predicted_ppm_temp = np.array([x['ppm'] for x in ppm_dict_list])
                    
                    predicted_ppm = np.vstack(([predicted_ppm,predicted_ppm_temp]))
            
            predicted_ppm = np.mean(predicted_ppm, axis = 0)
            
        return predicted_ppm
    except: 
        return None