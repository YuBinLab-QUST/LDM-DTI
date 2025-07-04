from os.path import exists
import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from torch.nn import SiLU
from Proteinencoder import CNNTransBlock
from DWFusion import BiIntention
from torch import Tensor
from typing import Tuple
from Proteinencoder import FeedForwardModule
from egnn_pytorch import EGNN
import time
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class LDMDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(LDMDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']


        self.drug_encoding = EGNN(dim = 128)
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinACmix(max_len=1200,encoder_dim=128,num_layers=3,num_attention_heads=8,feed_forward_expansion_factor=4,feed_forward_dropout_p=0.1,attention_dropout_p=0.1,conv_dropout_p=0.1,conv_kernel_size=3)

        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, feature_vectors,feature,coor,bg_d,v_p, protein_mask,mode="train"):
        v_d = self.drug_extractor(bg_d)#v_d.shape(64, 290, 128)
        drug_feature = self.drug_encoding(feature,coor)
        drug_feature_tensor = drug_feature[0]
        # 扩展 drug_feature_tensor 的第一个维度
        drug_feature_tensor = drug_feature_tensor.expand(v_d.shape[0], -1, -1)  # (32, 64, 128)
        # 将 v_d 和 drug_feature_tensor 拼接在第二个维度上
        v_d = torch.cat([v_d, drug_feature_tensor], dim=1)  #
        #v_d=torch.cat([v_d,feature_vectors],dim=1)
        v_d = 0.3 * feature_vectors + 0.7 * v_d#v_d.shape:(32,354,128)
        v_p = self.protein_extractor(v_p,protein_mask)#v_p.shape:(32, 1200, 128)
        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)#f:[64, 256]
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]



    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinACmix(nn.Module):
    def __init__(
            self,
            max_len: int = 1200,
            encoder_dim: int = 512,
            num_layers: int = 3,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            top_k: int = 5,
            k1: int = 3,
    ):
        super(ProteinACmix, self).__init__()
        self.CNNTranslayers = nn.ModuleList([CNNTransBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            top_k=top_k,
            k1=k1,
            max_len=max_len
        ) for _ in range(num_layers)])

        self.FFlayers = nn.ModuleList([FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        ) for _ in range(num_layers)])

    def forward(self, inputs: Tensor,mask:Tensor) -> Tuple[Tensor,Tensor]:
        #mask = (inputs.sum(dim=-1) != 0).float()
        CNNTransOutputs = inputs.to(dtype=torch.float32)
        for num in range(len(self.CNNTranslayers)):
            FF_output = 0.5 * self.FFlayers[num](CNNTransOutputs) + 0.5 * CNNTransOutputs
            CNNTransOutputs = self.CNNTranslayers[num](FF_output,mask)
        return CNNTransOutputs

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
