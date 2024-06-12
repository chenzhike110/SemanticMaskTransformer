import torch
import wandb
import random
import config
import torch.nn as nn
from modules.base import BaseModel
from torch_geometric.data import Batch, Data
from modules.graph.blocks import Encoder, Decoder
from modules.vq.residual_vq import GroupedResidualVQ
from modules.losses.geodesic import geodesic_loss_R
from libs.body_model.body_model import BodyModel
from tools.smpl_utils import pose_data_as_dict, get_smplh_groups
from tools.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle

class GRQVAE(BaseModel):
    def __init__(self, cfg):
        super(GRQVAE, self).__init__(cfg=cfg)
        self.input_dim = cfg.model.params.input_dim
        self.group_num = cfg.model.params.group_num
        self.output_dim = cfg.model.params.output_dim
        self.codebook_dim = cfg.model.params.codebook_dim
        self.joints_index, self.group_index = get_smplh_groups(cfg.model.params.group_num, unique=cfg.model.params.unique)
        
        self.quantizer_num = cfg.model.params.num_quantizers
        self.group_dim = cfg.model.params.group_dim
        self.num_nodes = cfg.model.params.num_nodes
        hidden_dims = cfg.model.params.hidden_dims
        self.hidden_dims = cfg.model.params.hidden_dims[-1]

        self.encoder = Encoder(
            in_channels=self.input_dim,
            hidden_dims=hidden_dims,
            out_channels=self.codebook_dim,
            dim=0,
        )

        self.decoder = Decoder(
            in_channels=self.codebook_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=self.output_dim,
            dim=0,
        )

        self.residual_vq = GroupedResidualVQ(
            dim = self.group_num * self.group_dim,
            num_quantizers = self.quantizer_num,      # specify number of quantizers
            groups = self.group_num,
            codebook_size = cfg.model.params.codebook_size,
            codebook_dim = self.codebook_dim,
            accept_image_fmap = False, # split on dimension 1
            split_dim = 1,
            use_cosine_sim=cfg.model.params.use_cosine_codebook,
            quantize_dropout=cfg.model.params.quantize_dropout
        )

        # Nest Modules
        self.fuzzy_modules = nn.ModuleList(
            [nn.Linear(len(self.group_index[i])*hidden_dims[-1], self.group_dim*hidden_dims[-1]) for i in range(self.group_num)]
        )
        
        decompose_index = [x for xx in self.group_index for x in xx]
        self.decompose_index = [decompose_index.index(i) for i in range(max(decompose_index)+1)]

        self.defuzzy_module = nn.ModuleList(
            [nn.Linear(self.group_dim*hidden_dims[-1], len(self.group_index[i])*hidden_dims[-1]) for i in range(self.group_num)]
        )

        self.bm = BodyModel(bm_fname=config.SMPLH_BODY_MODEL_PATH, num_betas=16)
        self.kinematic_tree = [[j, i] for i, j in enumerate(self.bm.kintree_table[0].long()) if j >= 0]
        self.kinematic_tree = [tree for tree in self.kinematic_tree if tree[0] in self.joints_index and tree[1] in self.joints_index]
        self.kinematic_tree = torch.tensor(self.kinematic_tree).long().permute(1, 0)

    @torch.no_grad()
    def preprocess(self, data):
        """
        data (NxJx3) rotvec
        """
        smpl = self.bm(**pose_data_as_dict(data))
        rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(data))
        data_seq = []
        for i in range(data.shape[0]):
            pose = Data(
                x=torch.cat([rot6d[i, self.joints_index], smpl.Jtr[i, self.joints_index]], dim=-1),
                edge_index=self.kinematic_tree
            )
            data_seq.append(pose)
        return Batch.from_data_list(data_seq)

    def compose_body_parts(self, x):
        x = x.permute(0, 2, 1)
        composition = []
        for index, group in enumerate(self.group_index):
            composition.append(self.fuzzy_modules[index](x[..., group].contiguous().flatten(start_dim=1)).reshape(x.shape[0], self.hidden_dims, self.group_dim))
        composition = torch.cat(composition, dim=-1).permute(0, 2, 1)
        return composition
    
    def decompose_body_parts(self, x):
        x = x.permute(0, 2, 1)
        assert x.shape[-1] == len(self.group_index)
        decomposition = []
        for index, group in enumerate(self.group_index):
            decomposition.append(self.defuzzy_module[index](x[..., index, None].contiguous().flatten(start_dim=1)).reshape(x.shape[0], self.hidden_dims, len(group)))
        decomposition = torch.cat(decomposition, dim=-1)[..., self.decompose_index].permute(0, 2, 1)
        return decomposition
    
    def forward(self, data):
        x = self.preprocess(data)
        x = self.encoder(x)
        x = self.compose_body_parts(x)
        quantized, indices, commit_loss = self.residual_vq(x)
        x = self.decompose_body_parts(quantized)
        x = self.decoder(x, data)

        # compute loss
        return x, indices, commit_loss
    
    def encode(self, data):
        x = self.preprocess(data)
        x = self.encoder(x)
        x = self.compose_body_parts(x)
        quantized, indices, commit_loss = self.residual_vq(x)
        return quantized, indices, commit_loss

    def decode(self, x, data):
        quantized, indices, commit_loss = self.residual_vq(x)
        x = self.decompose_body_parts(quantized)
        x = self.decoder(x, data)
        return x
    
    def decode_with_index(self, indices, data):
        quantized = self.residual_vq.get_output_from_indices(indices)
        x = self.decompose_body_parts(quantized)
        x = self.decoder(x, data)
        return x

    @classmethod
    def from_pretrained(cls, path):
        state_dict = torch.load(path)
        model = cls(state_dict['cfg'])
        model.load_state_dict(state_dict['state_dict'])
        return model
    
    def compute_rec_loss(self, pred, gt):
        l1_criterion = torch.nn.L1Loss()
        rot_criterion = geodesic_loss_R()

        v2v_loss = l1_criterion(pred.v, gt.v)
        j2j_loss = l1_criterion(pred.Jtr, gt.Jtr)

        rot_loss = rot_criterion(pred.rot_mat, gt.rot_mat).mean()

        return {
            'loss_v2v': v2v_loss,
            'loss_j2j': j2j_loss,
            'loss_rot': rot_loss,
        }
    
    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            smpl_ori = self.bm(**pose_data_as_dict(batch['pose']))
            smpl_ori.rot_mat = axis_angle_to_matrix(batch['pose'][:, self.joints_index])
        
        x, indices, commit_loss = self(batch['pose'])

        rot_mat = rotation_6d_to_matrix(x)
        rot_vec = matrix_to_axis_angle(rot_mat)

        batch['pose'][:, self.joints_index] = rot_vec

        smpl_rec = self.bm(**pose_data_as_dict(batch['pose']))
        smpl_rec.rot_mat = rot_mat

        loss_dict = self.compute_rec_loss(smpl_rec, smpl_ori)
        loss_dict['loss_com'] = commit_loss.mean()

        loss = 0
        for k, v in loss_dict.items():
            weight = self.cfg.train.loss.get('w'+k, 0.0)
            loss += weight * v
        loss_dict['loss'] = loss

        self.log_dict({f'train/{k}': v for k, v in loss_dict.items()}, batch_size=batch['pose'].shape[0])

        return {'loss': loss_dict['loss'] }
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        
        smpl_ori = self.bm(**pose_data_as_dict(batch['pose']))
        smpl_ori.rot_mat = axis_angle_to_matrix(batch['pose'][:, self.joints_index])
        
        x, indices, commit_loss = self(batch['pose'])

        rot_mat = rotation_6d_to_matrix(x)
        rot_vec = matrix_to_axis_angle(rot_mat)

        batch['pose'][:, self.joints_index] = rot_vec

        smpl_rec = self.bm(**pose_data_as_dict(batch['pose']))
        smpl_rec.rot_mat = rot_mat

        if batch_idx == 0 and self.cfg.train.visualize:
            index = random.randint(0, smpl_rec.v.shape[0]-1)
            vertices = torch.cat([smpl_rec.v[index, None].cpu(), smpl_ori.v[index, None].cpu()], dim=0)
            color = torch.arange(2)
            color = color.reshape(-1, 1, 1).repeat(1, smpl_rec.v.shape[1], 1)
            _3dPoints = torch.cat([vertices[..., [2,0,1]], color], dim=-1).view(-1, 4)
            self.logger.experiment.log({"Reconstructed Motion": [
                wandb.Object3D(_3dPoints.numpy())
            ]})

        loss_dict = self.compute_rec_loss(smpl_rec, smpl_ori)
        loss_dict['loss_com'] = commit_loss.mean()

        loss = 0
        for k, v in loss_dict.items():
            weight = self.cfg.train.loss.get('w'+k, 0.0)
            loss += weight * v
        loss_dict['loss'] = loss

        self.log_dict({f'val/{k}': v for k, v in loss_dict.items()}, batch_size=batch['pose'].shape[0])

        return {'loss': loss_dict['loss'] }