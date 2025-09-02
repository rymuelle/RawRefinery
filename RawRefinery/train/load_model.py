from Restorer.Restorer import Restorer, AddPixelShuffle
import torch

def make_model(width = 58, base_blocks = 2, dec_blocks=2, late_blocks=1, vit_blocks=0, expand_dims=2):

  total_late_block = max(base_blocks+late_blocks-vit_blocks, 0)
  enc_blks = [base_blocks, base_blocks, total_late_block, total_late_block]
  dec_blks = [dec_blocks, dec_blocks, dec_blocks, dec_blocks]
  vit_blks = [0, 0, vit_blocks, vit_blocks]
  middle_blk_num = base_blocks+late_blocks*2
  cond_output=32
  cond_input = 4
  drop_path = 0.0 #trial.suggest_float("drop_path",0, 0.1, log=False)
  drop_path_increment = 0.05 #trial.suggest_float("drop_path_increment",0, 0.1, log=False)
  model = Restorer
  model = Restorer(in_channels=4, out_channels=3 * 2 ** 2, width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, vit_blk_nums=vit_blks, dec_blk_nums=dec_blks,
                    cond_input = cond_input, cond_output=cond_output, expand_dims=expand_dims,
                   drop_path=drop_path,drop_path_increment=drop_path_increment)
  model = AddPixelShuffle(model)
  return model
#[FrozenTrial(number=0, state=1, values=[0.0010226645435831695, 2463.5001086660195],  params={'width': 80, 'expand_dims': 2, 'base_blocks': 1, 'dec_blks': 2, 'late_blocks': 0}, user_attrs={}, system_attrs={'fixed_params': {'width': 80, 'expand_dims': 2, 'base_blocks': 1, 'dec_blks': 2, 'late_blocks': 0}}, intermediate_values={}, distributions={'width': IntDistribution(high=90, log=False, low=30, step=1), 'expand_dims': IntDistribution(high=4, log=False, low=1, step=1), 'base_blocks': IntDistribution(high=3, log=False, low=1, step=1), 'dec_blks': IntDistribution(high=3, log=False, low=1, step=1), 'late_blocks': IntDistribution(high=5, log=False, low=0, step=1)}, trial_id=1, value=None),

def load_model(device, weight_file_path):
    #[FrozenTrial(number=4,  params={'width': 33, 'expand_dims': 2, 'base_blocks': 1, 'dec_blks': 2, 'late_blocks': 0}, user_attrs={}, system_attrs={'NSGAIISampler:generation': 0}, intermediate_values={}, distributions={'width': IntDistribution(high=90, log=False, low=30, step=1), 'expand_dims': IntDistribution(high=4, log=False, low=1, step=1), 'base_blocks': IntDistribution(high=3, log=False, low=1, step=1), 'dec_blks': IntDistribution(high=3, log=False, low=1, step=1), 'late_blocks': IntDistribution(high=5, log=False, low=0, step=1)}, trial_id=5, value=None),
    #FrozenTrial(number=12, state=1, values=[0.0012007935130358552, 574.8597779699994],params={'width': 90, 'expand_dims': 4, 'base_blocks': 1, 'dec_blks': 1, 'late_blocks': 0}, user_attrs={}, system_attrs={'NSGAIISampler:generation': 0}, intermediate_values={}, distributions={'width': IntDistribution(high=90, log=False, low=30, step=1), 'expand_dims': IntDistribution(high=4, log=False, low=1, step=1), 'base_blocks': IntDistribution(high=3, log=False, low=1, step=1), 'dec_blks': IntDistribution(high=3, log=False, low=1, step=1), 'late_blocks': IntDistribution(high=5, log=False, low=0, step=1)}, trial_id=13, value=None),
    width = 58
    expand_dims = 2
    base_blocks = 2
    dec_blocks = 2
    late_blocks = 1

    # Model, loss, optimizer
    model = make_model(width=width, base_blocks=base_blocks, late_blocks=late_blocks, expand_dims=expand_dims, dec_blocks=dec_blocks)
    model = model.to(device)
    state_dict = torch.load(weight_file_path,map_location=torch.device('cpu'))

    # new_dict = {}
    # for key, value in torch.load(weight_file_path,map_location=torch.device('cpu')).items():
    #     if 'conditioning_gen' not in key:
    #         new_dict[key] = value
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model
