import random
import torch
from unittest import TestCase
from unittest.mock import MagicMock, patch
from vllm.config import QuantizationConfig
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import GroupCoordinator as GroupCoordinatorGPU
from omni.models.config_loader.loader import model_extra_config
from omni.layers.moe.deepseek_moe import ReplicatedDeepseekMLP, ParallelDeepseekMLP
from omni.adaptors.vllm.distributed.parallel_state import GroupCoordinator


class test_ReplicatedDeepseekMLP(TestCase):
    @patch('vllm.platforms.current_platform')
    @patch('vllm.distributed.parallel_state._WORLD',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._PP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._EP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.config.QuantizationConfig',
    new_callable=lambda: MagicMock(spec=QuantizationConfig))    
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    def setUp(self,
              mock_model_extra_config,
              mock_quant_config,
              mock_vllm_ep,
              mock_vllm_pp,
              mock_vllm_world,
              mock_current_platform):
        self.mock_hidden_size = 7168
        self.mock_intermediate_size = 2048
        self.mock_hidden_act = "silu"
        self.mock_reduce_results = False
        self.mock_prefix = "model.layers.3.mlp.shared_experts"
        self.mock_ep_size = 8
        self.mock_world_size = 8
        self.mock_rank_in_group = 0
        self.mock_bsz = 256

        mock_model_extra_config.operator_opt_config.decode_moe_dispatch_combine = True
        mock_quant_config = None
        mock_current_platform.device_type = "npu"

        mock_vllm_ep.world_size = self.mock_ep_size
        mock_vllm_ep.rank_in_group = MagicMock()
        mock_vllm_ep.device_group = MagicMock()  
        mock_vllm_pp.world_size = self.mock_world_size
        mock_vllm_pp.rank_in_group = self.mock_rank_in_group
        mock_vllm_pp.device_group = MagicMock()  
        mock_vllm_world.world_size = self.mock_world_size
        mock_vllm_world.rank_in_group = self.mock_rank_in_group
        mock_vllm_world.device_group = MagicMock()  

        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._TP = mock_vllm_ep
        vllm_ps._PP = mock_vllm_pp
        vllm_ps._WORLD = mock_vllm_world

        self.mlp = ReplicatedDeepseekMLP(hidden_size=self.mock_hidden_size,
                                         intermediate_size=self.mock_intermediate_size,
                                         hidden_act=self.mock_hidden_act,
                                         quant_config=mock_quant_config,
                                         reduce_results=self.mock_reduce_results,
                                         prefix=self.mock_prefix)   
    
    def tearDown(self):   
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._TP = None
        vllm_ps._PP = None
        vllm_ps._WORLD = None   

    def test_initialization(self):
        self.assertIsNotNone(self.mlp.gate_up_proj)      
        self.assertIsNotNone(self.mlp.down_proj)     
        self.assertIsNotNone(self.mlp.act_fn_obj)      

        self.assertTrue(hasattr(self.mlp, 'ep_size'))
        self.assertTrue(hasattr(self.mlp, 'global_rank'))
        self.assertTrue(hasattr(self.mlp, 'world_size'))
        self.assertTrue(hasattr(self.mlp, 'moe_all_to_all_group'))
        self.assertTrue(hasattr(self.mlp, 'moe_all_to_all_group_name'))
        self.assertTrue(hasattr(self.mlp, 'moe_rs_group'))
        self.assertTrue(hasattr(self.mlp, 'moe_rs_group_rank'))
        self.assertTrue(hasattr(self.mlp, 'moe_rs_group_name'))

        self.assertEqual(self.mlp.tp_size, 1)
        self.assertEqual(self.mlp.ep_size, self.mock_ep_size)
        self.assertEqual(self.mlp.global_rank, self.mock_rank_in_group)
        self.assertEqual(self.mlp.world_size, self.mock_world_size)

    def test_act_fn(self):
        mock_input = [torch.randint(low=0, high=self.mock_bsz, size=(self.mock_bsz, 4096), dtype=torch.int32),
                      torch.randn(self.mock_bsz, dtype=torch.float32)]
        mock_quant_symbol = True

        self.mlp.gate_up_proj.weight_scale = MagicMock(return_value=torch.randn(4096, dtype=torch.float32))
        self.mlp.act_fn_obj.forward = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz, 2048), dtype=torch.int8),
                                                                "pertoken_scale":torch.randn(self.mock_bsz, dtype=torch.float32)})

        result = self.mlp.act_fn(x=mock_input, quant_symbol=mock_quant_symbol)

        self.assertIsInstance(result, dict)
        self.assertIn('x_int8', result)
        self.assertIn('pertoken_scale', result)
        self.assertIsInstance(result['x_int8'], torch.Tensor)
        self.assertIsInstance(result['pertoken_scale'], torch.Tensor)

    def test_forward(self):
        mock_input = torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16)
        self.mlp.quant_symbol = True
        
        self.mlp.gate_up_proj.forward = MagicMock(return_value=[torch.randint(low=0, high=self.mock_bsz, size=(self.mock_bsz, 4096), dtype=torch.int32),
                                                    torch.randn(self.mock_bsz, dtype=torch.float32)])
        self.mlp.act_fn = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz, 2048), dtype=torch.int8),
                                                          "pertoken_scale":torch.randn(self.mock_bsz, dtype=torch.float32)})
        self.mlp.down_proj.forward = MagicMock(return_value=[torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16), None])

        result = self.mlp.forward(x=mock_input)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], self.mock_bsz)
        self.assertEqual(result.shape[1], self.mock_hidden_size)


class test_ParallelDeepseekMLP(TestCase):
  
    @patch('omni.adaptors.vllm.distributed.parallel_state._MLP_TP',
    new_callable=lambda: MagicMock(spec=GroupCoordinator))    
    @patch('vllm.config.QuantizationConfig',
    new_callable=lambda: MagicMock(spec=QuantizationConfig))
    def setUp(self, mock_quant_config, mock_omni_mlp_tp):
        self.mock_hidden_size = 7168
        self.mock_intermediate_size = 18432
        self.mock_hidden_act = "silu"
        self.mock_reduce_results = True
        self.mock_prefix = "model.layers.0.mlp"
        self.mock_bsz = 8
        self.mock_world_size = 8
        self.mock_rank_in_group = 0

        mock_quant_config = None

        mock_omni_mlp_tp.world_size = self.mock_world_size
        mock_omni_mlp_tp.rank_in_group = 0
        mock_omni_mlp_tp.device_group = MagicMock()  

        import omni.adaptors.vllm.distributed.parallel_state as omni_ps
        omni_ps._MLP_TP = mock_omni_mlp_tp

        self.mlp = ParallelDeepseekMLP(hidden_size=self.mock_hidden_size,
                                       intermediate_size=self.mock_intermediate_size,
                                       hidden_act=self.mock_hidden_act,
                                       quant_config=mock_quant_config,
                                       reduce_results=self.mock_reduce_results,
                                       prefix=self.mock_prefix,
                                       comm_group=mock_omni_mlp_tp)

    def tearDown(self):   
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._MLP_TP = None

    def test_initialization(self):
        self.assertIsNotNone(self.mlp.gate_up_proj)      
        self.assertIsNotNone(self.mlp.down_proj)     
        self.assertIsNotNone(self.mlp.act_fn_obj)      

        self.assertTrue(hasattr(self.mlp, 'prefix'))
        self.assertTrue(hasattr(self.mlp, 'quant_symbol'))
        self.assertTrue(hasattr(self.mlp, 'comm_group'))

        self.assertEqual(self.mlp.comm_group.rank_in_group, self.mock_rank_in_group)
        self.assertEqual(self.mlp.comm_group.world_size, self.mock_world_size)

    def test_act_fn(self):
        mock_input = [torch.randint(low=0, high=self.mock_bsz, size=(self.mock_bsz, 4096), dtype=torch.int32),
                      torch.randn(self.mock_bsz, dtype=torch.float32)]
        mock_quant_symbol = True

        self.mlp.gate_up_proj.weight_scale = MagicMock(return_value=torch.randn(4096, dtype=torch.float32))
        self.mlp.act_fn_obj.forward = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz, 2048), dtype=torch.int8),
                                                                "pertoken_scale":torch.randn(self.mock_bsz, dtype=torch.float32)})
        result = self.mlp.act_fn(x=mock_input, quant_symbol=mock_quant_symbol)

        self.assertIsInstance(result, dict)
        self.assertIn('x_int8', result)
        self.assertIn('pertoken_scale', result)
        self.assertIsInstance(result['x_int8'], torch.Tensor)
        self.assertIsInstance(result['pertoken_scale'], torch.Tensor)

    def test_forward(self):
        self.mock_bsz_gather = self.mock_world_size * self.mock_bsz
        mock_input = torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16)
        self.mlp.quant_symbol = True     

        self.mlp.comm_group.all_gather = MagicMock(side_effect=lambda data, dim: data.repeat(self.mock_world_size, 1))

        self.mlp.gate_up_proj.forward = MagicMock(return_value=[torch.randint(low=0, high=self.mock_bsz_gather, size=(self.mock_bsz_gather, 4608), dtype=torch.int32),
                                                    torch.randn(self.mock_bsz_gather, dtype=torch.float32)])
        self.mlp.act_fn = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz_gather, 2304), dtype=torch.int8),
                                                  "pertoken_scale":torch.randn(self.mock_bsz_gather, dtype=torch.float32)})
        self.mlp.down_proj.forward = MagicMock(return_value=[torch.randn(self.mock_bsz_gather, self.mock_hidden_size, dtype=torch.bfloat16), None])

        self.mlp.comm_group.reduce_scatter = MagicMock(return_value=torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16))

        result_1 = self.mlp.forward(x=mock_input, residual=None, attn_metadata=None, layerid=None)           

        self.assertIsInstance(result_1, torch.Tensor)
        self.assertEqual(result_1.shape[0], self.mock_bsz)
        self.assertEqual(result_1.shape[1], self.mock_hidden_size)


        mock_residual = return_value=torch.randn(self.mock_bsz_gather, self.mock_hidden_size, dtype=torch.bfloat16)
        result_2, residual = self.mlp.forward(x=mock_input, residual=mock_residual, attn_metadata=None, layerid=None)           

        self.assertIsInstance(result_2, torch.Tensor)
        self.assertEqual(result_2.shape[0], self.mock_bsz)
        self.assertEqual(result_2.shape[1], self.mock_hidden_size)

        self.assertIsInstance(residual, torch.Tensor)
        self.assertEqual(residual.shape[0], self.mock_bsz_gather)
        self.assertEqual(residual.shape[1], self.mock_hidden_size)       


if __name__ == "__main__":
    unittest.main()