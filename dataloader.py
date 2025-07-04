import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from functools import lru_cache
import time
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
def one_of_k_encoding_unk(x, allowable_set):
    """
    将输入 x 编码为 one-hot 向量。如果 x 不在允许集合中，则将其编码为最后一个元素（'Unknown'）。
    """
    if x not in allowable_set:
        x = 'Unknown'
    return [x == s for s in allowable_set]

def one_of_k_encoding(x, allowable_set):
    """
    将输入 x 编码为 one-hot 向量。
    """
    return [1 if x == s else 0 for s in allowable_set]
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                                          'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
                                          'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
# 在函数定义前添加装饰器，缓存函数的调用结果
@lru_cache(maxsize=128)  # 这里设置缓存大小为128，可以根据实际情况调整
def get_coors(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES:{smile}")
        # 加氢
        mol = Chem.AddHs(mol)
        # 尝试生成三维构像
        num_confs = 3
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
        # 检查是否成功嵌入
        if len(ids) == 0:
            raise ValueError(f"Embedding failed for SMILES:{smile}")
        # 优化第一个有效构像
        if AllChem.UFFOptimizeMolecule(mol, confId=ids[0]) == -1:
            raise ValueError(f"UFF optimization failed for SMILES:{smile}")
        c_size = mol.GetNumAtoms()
        # 原子特征
        features = []
        coordinates = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            feature = atom_features(atom)
            features.append(feature / sum(feature))
            #获取原子特征坐标并转换为numpy数组
            pos=mol.GetConformer().GetAtomPosition(atom_idx)
            coordinates.append(np.array(pos))
        #return feature,coordinates
        return np.array(features), np.array(coordinates)
    except Exception as e:
        # 返回默认值，确保调用者可以继续处理
        return np.zeros((0,)), np.zeros((0, 3))  # 返回空特征和坐标
# 外部初始化模型
model_name = "/root/LDM-DTI-main/LDMDTI/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

path="/root/LDM-DTI-main/LDM-DTI/prot_bert"
prot_tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
prot_encoder = BertModel.from_pretrained(path)
class DTIDataset(data.Dataset):

    def __init__(self, list_IDs,  df, max_drug_nodes=290):
        self.linear_layer = nn.Linear(1024, 128)
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.tokenizer = tokenizer
        self.model = model
        self.prot_tokenizer = prot_tokenizer
        self.prot_encoder = prot_encoder
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.p_linear = nn.Linear(1,128)
        # 批量处理药物分子编码
        self.all_smiles = self.df['SMILES'].tolist()
        tokenized_inputs = self.tokenizer(self.all_smiles, return_tensors="pt", padding=True)
        all_outputs_list = []
        with torch.no_grad():
            for i in range(int(len(tokenized_inputs['input_ids'])/1000+1)):
                all_outputs_list.append(self.model(tokenized_inputs['input_ids'][i*1000:(i+1)*1000]).logits)
            # self.all_feature_vectors = self.all_outputs.mean(dim=1)
        self.all_outputs = torch.cat(all_outputs_list,dim=0)
        self.all_feature_vectors = self.all_outputs.mean(dim=1)
        # with torch.no_grad():
        #     self.all_outputs = self.model(**tokenized_inputs).logits
        #     self.all_feature_vectors = self.all_outputs.mean(dim=1)

        # 批量处理蛋白质编码
        self.all_proteins = self.df['Protein'].tolist()
        protein_inputs = self.prot_tokenizer(self.all_proteins, return_tensors='pt', padding=True, truncation=True,
                                             max_length=1200)
        with torch.no_grad():
            self.all_protein_features = self.prot_encoder(**protein_inputs).last_hidden_state.mean(dim=1)
            self.all_protein_features = torch.Tensor(self.all_protein_features).to(torch.float32)
            self.all_protein_features = self.linear_layer(self.all_protein_features)
    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def __getitem__(self, index):
        index = self.list_IDs[index]
        feature_vectors = self.all_feature_vectors[index].unsqueeze(0)  # 获取对应索引的编码结果
        v_d = self.df.iloc[index]['SMILES']
        #获取药物分子原子的三位坐标
        feature,coor = get_coors(v_d)
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        v_p = self.df.iloc[index]['Protein']
        pro_len = len(v_p)
        y = self.df.iloc[index]["Y"]
        protein_max = 1200
        protein_mask = np.zeros(protein_max)
        v_p = self.all_protein_features[index].unsqueeze(0).repeat(1200, 1)  # 获取对应索引的编码结果并重复
        if pro_len > protein_max:
            protein_mask[:] = 1
        else:
            protein_mask[:pro_len] = 1
        return feature_vectors,feature,coor,v_d, v_p, protein_mask, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
