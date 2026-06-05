#dram = #ttnn.buffer_type<dram>
#loc1 = loc("x")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <blackhole>, grid = 10x11, coord_translation_offsets = 2x1, l1_size = 1572864, num_dram_channels = 8, dram_channel_size = 4278190080, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 64, noc_dram_address_align_bytes = 64, l1_unreserved_base = 111104, erisc_l1_unreserved_base = 87872, dram_unreserved_base = 4800064, dram_unreserved_end = 4276412416, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x131072xf32, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4096x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @jit_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, ttcore.meshes = #ttcore.meshes<[<"empty_mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @jit_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, ttcore.meshes = #ttcore.meshes<[<"empty_mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<10x11, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 8, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 8) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func public @main(%arg0: tensor<131072xf32, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<131072xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>} loc("x")) -> (tensor<f32, #ttnn_layout1> {jax.result_info = "result", ttcore.local_shape = #ttcore<local_shape local_shape = tensor<f32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<131072xf32, #ttnn_layout>) -> tensor<131072xf32, #ttnn_layout2> loc(#loc7)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<131072xf32, #ttnn_layout>) -> () loc(#loc7)
        %1 = "ttnn.multiply"(%0, %0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<131072xf32, #ttnn_layout2>, tensor<131072xf32, #ttnn_layout2>) -> tensor<131072xf32, #ttnn_layout2> loc(#loc8)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<131072xf32, #ttnn_layout2>) -> () loc(#loc8)
        %2 = "ttnn.sum"(%1) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true, packer_l1_acc = true>, dim_arg = [0 : i32], keep_dim = false}> : (tensor<131072xf32, #ttnn_layout2>) -> tensor<f32, #ttnn_layout1> loc(#loc9)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<131072xf32, #ttnn_layout2>) -> () loc(#loc9)
        %3 = "ttnn.sqrt"(%2) : (tensor<f32, #ttnn_layout1>) -> tensor<f32, #ttnn_layout1> loc(#loc10)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<f32, #ttnn_layout1>) -> () loc(#loc10)
        return %3 : tensor<f32, #ttnn_layout1> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/houjun/.agents/embed_bw_min.py":26:17 to :63)
#loc3 = loc("/home/houjun/.agents/embed_bw_min.py":34:36 to :41)
#loc4 = loc("gn"(#loc2))
#loc5 = loc("<module>"(#loc3))
#loc6 = loc(callsite(#loc4 at #loc5))
#loc7 = loc("jit(norm)/mul_in_0_layout"(#loc6))
#loc8 = loc("jit(norm)/mul"(#loc6))
#loc9 = loc("jit(norm)/reduce_sum"(#loc6))
#loc10 = loc("jit(norm)/sqrt"(#loc6))
