from omegaconf import OmegaConf

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    # 将普通字典转换为 OmegaConf 对象，以支持点号访问属性
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    model = models[name](config)
    return model


from . import nerf, neus, geometry, texture
