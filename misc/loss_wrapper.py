import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from models import VSEFCModel

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.struc_crit = utils.StructureLosses(opt)


        if opt.vse_model != 'None':
            self.vse = VSEFCModel(opt)
            for p in self.vse.parameters():
                p.requires_grad = False
            self.retrieval_reward_weight = opt.retrieval_reward_weight # 

            self.vse.load_state_dict({k[4:]:v for k,v in torch.load(opt.initialize_retrieval).items() if 'vse.' in k})
        self.retrieval_reward_weight = 0

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag):
        opt = self.opt
        
        out = {}

        reduction = 'none' if drop_worst_flag else 'mean'
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:], reduction=reduction)
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                opt={'sample_max':0,
                    'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                        or not 'margin' in opt.structure_loss_type,
                    'sample_n': opt.structure_sample_n},
                mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            struc_loss = self.struc_crit(sample_logprobs, gen_result, gts, reduction=reduction)
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:], reduction=reduction)        
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
                if self.retrieval_reward_weight > 0:
                    _seqs_greedy, _sampleLogProbs_greedy = greedy_res, _
                    _masks_greedy = torch.cat([_seqs_greedy.data.new(_seqs_greedy.size(0), 2).fill_(1).float(), (_seqs_greedy > 0).float()[:, :-1]], 1)
                    _seqs_greedy = torch.cat([_seqs_greedy.data.new(_seqs_greedy.size(0), 1).fill_(self.model.vocab_size + 1), _seqs_greedy], 1)

                    baseline = self.vse(fc_feats, att_feats, att_masks, _seqs_greedy, _masks_greedy, True, only_one_retrieval='off')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            out['reward'] = reward[:,0].mean()

            if self.retrieval_reward_weight > 0:
                _seqs, _sampleLogProbs = gen_result, sample_logprobs
                _masks = torch.cat([_seqs.data.new(_seqs.size(0), 2).fill_(1).float(), (_seqs > 0).float()[:, :-1]], 1)

                gen_masks = _masks

                _seqs = torch.cat([_seqs.data.new(_seqs.size(0), 1).fill_(self.model.vocab_size + 1), _seqs], 1)

                retrieval_loss = self.vse(fc_feats, att_feats, att_masks, _seqs, _masks, True, only_one_retrieval='off')
        
                reward -= self.retrieval_reward_weight * (retrieval_loss - baseline).unsqueeze(1)

                out['retrieval_loss'] = retrieval_loss.sum()
                out['retrieval_loss_greedy'] = baseline.sum()

                print(out['retrieval_loss'].item(), out['retrieval_loss_greedy'].item())
	
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)

        out['loss'] = loss
        return out
