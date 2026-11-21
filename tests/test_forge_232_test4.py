"""NVFP4 numerics and GGUF reload behavior with actual CPU tensors.

AST loading isolates import-time device setup; no CUDA kernel is exercised.
"""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import gguf
import numpy as np
import torch

from test_vae_flash_regressions import load_nodes

ROOT = Path(__file__).resolve().parents[1]


def operations():
    namespace = dict(torch=torch, ForgeOperations=object, ForgeWeights=type('ForgeWeights', (), {}),
        current_device=torch.device('cpu'), current_dtype=torch.float16,
        memory_management=types.SimpleNamespace(supports_nvfp4_compute=lambda device:False,
                                                get_torch_device=lambda:torch.device('cpu')))
    utility = load_nodes('backend/utils.py', ['tensor2parameter'], {'torch':torch})
    namespace['utils'] = types.SimpleNamespace(**utility)
    tree=ast.parse((ROOT/'backend/operations.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ForgeOperationsGGUF')
    exec(compile(ast.Module(body=[node],type_ignores=[]), 'backend/operations.py', 'exec'),namespace)
    return namespace['ForgeOperationsGGUF']


class GGUFReloadTests(unittest.TestCase):
    def test_linear_conv_and_embedding_parameters_remain_registered_after_reload(self):
        ops=operations()
        for factory,shape in ((lambda:ops.Linear(),(2,3)),(lambda:ops.Conv2d(1,2,1),(2,1,1,1)),
                              (lambda:ops.Embedding(4,3),(4,3))):
            with self.subTest(shape=shape):
                model=factory()
                for offset in (0,10):
                    weight=torch.arange(np.prod(shape)).reshape(shape).float()+offset
                    model.load_state_dict({'weight':weight},strict=False)
                    self.assertIn('weight',dict(model.named_parameters()))
                    torch.testing.assert_close(model.weight,weight)
                    self.assertEqual(model._computation_dtype,torch.float16)

    def test_standard_meta_layer_dequantizes_while_gguf_layer_retains_packed_weight(self):
        from backend.operations_gguf import ParameterGGUF,dequantize_tensor
        from backend.state_dict import load_state_dict
        ops=operations()
        raw=np.zeros((2,18),dtype=np.uint8)
        raw[:,:2]=np.array([1],dtype=np.float16).view(np.uint8)
        raw[:,2:]=0x98
        def parameter():
            return ParameterGGUF(torch.from_numpy(raw.copy()),tensor_type=gguf.GGMLQuantizationType.Q4_0,
                                 tensor_shape=torch.Size((2,32)))
        module=types.ModuleType('backend.operations');module.ForgeOperationsGGUF=ops
        with patch.dict(sys.modules,{'backend.operations':module}):
            standard=torch.nn.Linear(32,2,bias=False,device='meta')
            with redirect_stdout(io.StringIO()):load_state_dict(standard,{'weight':parameter()})
            expected=gguf.quants.dequantize(raw,gguf.GGMLQuantizationType.Q4_0)
            np.testing.assert_array_equal(standard.weight.detach().numpy(),expected)
            quantized=ops.Linear()
            quantized.load_state_dict({'weight':parameter()})
            self.assertIsInstance(quantized.weight,ParameterGGUF)
            with redirect_stdout(io.StringIO()):load_state_dict(quantized,{'weight':parameter()})
            self.assertIsInstance(quantized.weight,ParameterGGUF)
            quantized.weight.gguf_cls.bake(quantized.weight)
            np.testing.assert_array_equal(dequantize_tensor(quantized.weight).float().numpy(),expected)


class NVFP4Tests(unittest.TestCase):
    def test_all_finite_scale_codes_and_signed_nibbles_match_independent_decoder(self):
        from backend.operations_gguf import ParameterGGUF,dequantize_tensor
        for scale in range(128):
            raw=np.full((1,36),0x98,dtype=np.uint8)
            raw[:,:4]=scale
            raw[:,4:]=np.arange(32,dtype=np.uint8)*7
            parameter=ParameterGGUF(torch.from_numpy(raw.copy()),tensor_type=gguf.GGMLQuantizationType.NVFP4,
                                    tensor_shape=torch.Size((1,64)))
            parameter.computation_dtype=torch.float32
            parameter.gguf_cls.bake(parameter)
            expected=gguf.quants.dequantize(raw,gguf.GGMLQuantizationType.NVFP4)
            np.testing.assert_allclose(dequantize_tensor(parameter).numpy(),expected,rtol=0,atol=0)

    def test_pin_memory_keeps_quantization_metadata(self):
        from backend.operations_gguf import ParameterGGUF
        parameter=ParameterGGUF(torch.zeros((2,36),dtype=torch.uint8),tensor_type=gguf.GGMLQuantizationType.NVFP4,
                                tensor_shape=torch.Size((2,64)))
        # Actual pinned allocation requires an accelerator. Isolate that allocation only.
        with patch.object(torch.Tensor,'pin_memory',lambda tensor,device=None:tensor.clone()):
            pinned=parameter.pin_memory()
        self.assertEqual(pinned.shape,parameter.shape)
        self.assertIs(pinned.gguf_cls,parameter.gguf_cls)
        self.assertEqual(pinned.tensor_type,parameter.tensor_type)
        self.assertFalse(pinned.baked)
        self.assertNotEqual(pinned.data_ptr(),parameter.data_ptr())
