from dotmap import DotMap
import os
import yaml

def load_config(default_ps_fname=None, **kwargs):
    if isinstance(default_ps_fname, str):
        assert os.path.exists(default_ps_fname), FileNotFoundError(default_ps_fname)
        assert default_ps_fname.lower().endswith('.yaml'), NotImplementedError('Only .yaml files are accepted.')
        default_ps = yaml.safe_load(open(default_ps_fname, 'r'))
    else:
        default_ps = {}

    default_ps.update(kwargs)

    return DotMap(default_ps)

def dump_config(data, fname):
    '''
    dump current configuration to an ini file
    :param fname:
    :return:
    '''
    with open(fname, 'w') as file:
        yaml.dump(data.toDict(), file)
    return fname

# a = load_config('/is/ps3/nghorbani/code-repos/supercap/support_data/supercap_defaults.yaml')
#
# print(a)
#{**a.train_params.optimizer.args, 'lr': 1e-15}
# a.pprint()