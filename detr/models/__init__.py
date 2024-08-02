# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build_vae as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp

def build_ACT_model_(args):
    return build_vae(args)

def build_CNNMLP_model_(args):
    return build_cnnmlp(args)