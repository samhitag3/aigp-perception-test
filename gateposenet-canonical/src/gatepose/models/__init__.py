from gatepose.models.gateposenet import GatePoseNetCanonical

def build_model(cfg:dict): return GatePoseNetCanonical(cfg)
