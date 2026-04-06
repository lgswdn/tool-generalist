#!/usr/bin/env python3
"""
Test script for ActorCriticICP module
"""


import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_icp import ActorCriticICP
from rsl_rl.modules.models.rl.net.icp import ICPNet


def test_actor_critic_icp():
    """
    Test function to verify ActorCriticICP functionality
    """
    print("=" * 60)
    print("Testing ActorCriticICP Module")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    num_points = 512
    icp_point_dim = 3
    hand_state_dim = 9
    other_obs_dim = 32  # Other regular observations
    
    # Calculate total observation dimension: point_cloud + hand_state + other_obs
    num_actor_obs = num_points * icp_point_dim + hand_state_dim + other_obs_dim
    num_critic_obs = num_actor_obs
    num_actions = 12
    
    # Create test observations as flattened tensor
    point_cloud_flat = torch.randn(batch_size, num_points * icp_point_dim)
    hand_state = torch.randn(batch_size, hand_state_dim)
    other_obs = torch.randn(batch_size, other_obs_dim)
    
    # Concatenate all observations
    observations = torch.cat([point_cloud_flat, hand_state, other_obs], dim=-1)
    
    print(f"Input shapes:")
    print(f"  observations: {observations.shape}")
    print(f"  point_cloud (reshaped): {point_cloud_flat.view(batch_size, num_points, icp_point_dim).shape}")
    print(f"  hand_state: {hand_state.shape}")
    print(f"  other_obs: {other_obs.shape}")
    
    # Test 1: Initialize ActorCriticICP with pretrained weights
    print("\n" + "-" * 40)
    print("Test 1: Initialize with pretrained weights")
    print("-" * 40)
    
    model = ActorCriticICP(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        icp_point_dim=icp_point_dim,
        icp_num_points=num_points,
        icp_weights_path='/home/steve/corn/ckpts/512-32-balanced-SAM-wd-5e-05-920',  # No pretrained weights
        freeze_icp=True,  # Allow training
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        fusion_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0
    )
    print("✓ Model initialized successfully!")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        

    
    # Test 2: Forward pass - Actor inference
    print("\n" + "-" * 40)
    print("Test 2: Actor inference (deterministic)")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        actions_mean = model.act_inference(observations)
    
    print(f"✓ Actor inference successful!")
    print(f"  Actions shape: {actions_mean.shape}")
    print(f"  Actions mean: {actions_mean.mean().item():.4f}")
    print(f"  Actions std: {actions_mean.std().item():.4f}")
    
    # Test 3: Forward pass - Actor sampling
    print("\n" + "-" * 40)
    print("Test 3: Actor sampling (stochastic)")
    print("-" * 40)
    
    try:
        model.train()
        actions = model.act(observations)
        
        print(f"✓ Actor sampling successful!")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Actions mean: {actions.mean().item():.4f}")
        print(f"  Actions std: {actions.std().item():.4f}")
        
        # Test log probabilities
        log_probs = model.get_actions_log_prob(actions)
        print(f"  Log probs shape: {log_probs.shape}")
        print(f"  Log probs mean: {log_probs.mean().item():.4f}")
        
    except Exception as e:
        print(f"✗ Actor sampling failed: {e}")
        return False
    
    # Test 4: Critic evaluation
    print("\n" + "-" * 40)
    print("Test 4: Critic evaluation")
    print("-" * 40)
    
    try:
        values = model.evaluate(observations)
        
        print(f"✓ Critic evaluation successful!")
        print(f"  Values shape: {values.shape}")
        print(f"  Values mean: {values.mean().item():.4f}")
        print(f"  Values std: {values.std().item():.4f}")
        
    except Exception as e:
        print(f"✗ Critic evaluation failed: {e}")
        return False
    
    # Test 5: Action distribution properties
    print("\n" + "-" * 40)
    print("Test 5: Action distribution properties")
    print("-" * 40)
    
    try:
        model.update_distribution(observations)
        
        action_mean = model.action_mean
        action_std = model.action_std
        entropy = model.entropy
        
        print(f"✓ Distribution properties computed successfully!")
        print(f"  Action mean shape: {action_mean.shape}")
        print(f"  Action std shape: {action_std.shape}")
        print(f"  Entropy shape: {entropy.shape}")
        print(f"  Mean entropy: {entropy.mean().item():.4f}")
        
    except Exception as e:
        print(f"✗ Distribution properties failed: {e}")
        return False
    
    # Test 6: Gradient computation
    print("\n" + "-" * 40)
    print("Test 6: Gradient computation")
    print("-" * 40)
    
    try:
        model.train()
        model.zero_grad()
        
        # Forward pass
        actions = model.act(observations)
        values = model.evaluate(observations)
        log_probs = model.get_actions_log_prob(actions)
        
        # Compute a simple loss
        action_loss = actions.mean()
        value_loss = values.mean()
        policy_loss = -log_probs.mean()
        
        total_loss = action_loss + value_loss + policy_loss
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0.0
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        grad_norm = grad_norm ** 0.5
        
        print(f"✓ Gradient computation successful!")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Gradient norm: {grad_norm:.4f}")
        print(f"  Parameters with gradients: {param_count}")
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    # Test 7: Feature extraction
    print("\n" + "-" * 40)
    print("Test 7: Feature extraction")
    print("-" * 40)
    
    try:
        # Test point cloud and context extraction
        point_cloud, context, regular_obs = model._extract_point_cloud_and_context(observations)
        
        print(f"✓ Feature extraction successful!")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Context keys: {list(context.keys())}")
        print(f"  Regular obs shape: {regular_obs.shape}")
        
        # Test fused features
        fused_features = model._get_fused_features(observations)
        print(f"  Fused features shape: {fused_features.shape}")
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False
    
    # Test 8: Model state management
    print("\n" + "-" * 40)
    print("Test 8: Model state management")
    print("-" * 40)
    
    try:
        # Test train/eval mode switching
        model.train()
        train_mode = model.training
        
        model.eval()
        eval_mode = not model.training
        
        # Test reset (should be no-op)
        model.reset()
        
        print(f"✓ Model state management successful!")
        print(f"  Train mode works: {train_mode}")
        print(f"  Eval mode works: {eval_mode}")
        print(f"  Reset method works: True")
        
    except Exception as e:
        print(f"✗ Model state management failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ActorCriticICP is working correctly.")
    print("=" * 60)
    
    return True


def test_with_different_inputs():
    """
    Test ActorCriticICP with different input configurations
    """
    print("\n" + "=" * 60)
    print("Testing with different input configurations")
    print("=" * 60)
    
    configurations = [
        {
            "name": "Small config",
            "batch_size": 2,
            "num_points": 256,
            "icp_point_dim": 3,
            "other_obs_dim": 32,
            "actor_hidden_dims": [128, 64],
            "fusion_hidden_dims": [256, 128],
        },
        {
            "name": "Large config", 
            "batch_size": 8,
            "num_points": 1024,
            "icp_point_dim": 3,
            "other_obs_dim": 128,
            "actor_hidden_dims": [512, 256, 128],
            "fusion_hidden_dims": [1024, 512, 256],
        }
    ]
    
    for i, config in enumerate(configurations):
        print(f"\n{'-'*20} {config['name']} {'-'*20}")
        
        # Create observations
        point_cloud_flat = torch.randn(config['batch_size'], config['num_points'] * 3)
        hand_state = torch.randn(config['batch_size'], 9)
        other_obs = torch.randn(config['batch_size'], config['other_obs_dim'])
        num_actor_obs = config['num_points'] * 3 + 9 + config['other_obs_dim']

        observations = torch.cat([point_cloud_flat, hand_state, other_obs], dim=-1)
        
        # Create model
        model = ActorCriticICP(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_actor_obs,
            num_actions=6,
            icp_point_dim=config['icp_point_dim'],
            icp_num_points=config['num_points'],
            icp_weights_path=None,
            freeze_icp=True,
            context_keys=["hand_state"],
            actor_hidden_dims=config['actor_hidden_dims'],
            critic_hidden_dims=config['actor_hidden_dims'],
            fusion_hidden_dims=config['fusion_hidden_dims'],
            activation="elu",
            init_noise_std=1.0
        )
        
        # Quick test
        actions = model.act(observations)
        values = model.evaluate(observations)
        
        print(f"✓ {config['name']} successful!")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Values shape: {values.shape}")
        print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

        # 测试配置匹配
        print("\n" + "="*60)
        print("Testing ICP configuration matching...")
        print("="*60)
        
        # 测试错误的配置 - 创建一个真正错误的配置
        print("Testing with WRONG configuration (should fail):")
        try:
            # 创建一个错误的ICP配置来测试strict=True
            wrong_cfg = ICPNet.Config(
                dim_in=(config['num_points'], config['icp_point_dim']),
                dim_out=256,  # 错误的输出维度
                keys={'hand_state': 9},
                headers=['collision'],
                num_query=1,
                patch_size=32,
                encoder_channel=256,  # 错误的编码器维度
                pos_embed_type='mlp',
                group_type='fps',
                patch_type='mlp',
                patch_overlap=1.0,
                p_drop=0.0,
                freeze_encoder=False,
                use_adapter=False,
                adapter_dim=64,
                tune_last_layer=False,
                late_late_fusion=False,
                output_attn=False,
                output_hidden=False,
                activate_header=False,
                pre_ln_bias=True,
                ignore_zero=False,
                use_vq=False,
                train_last_ln=True,
                header_inputs=None,
                use_v2_module=False
            )
            
            # 设置错误的编码器配置
            wrong_cfg.encoder.num_hidden_layers = 4  # 错误的层数
            wrong_cfg.encoder.layer.hidden_size = 256  # 错误的隐藏维度
            wrong_cfg.encoder.layer.num_attention_heads = 8  # 错误的头数
            
            # 创建错误的ICP模型
            wrong_icp = ICPNet(wrong_cfg)
            
            # 尝试加载权重 - 这应该失败
            wrong_icp.load(filename='/home/steve/corn/ckpts/512-32-balanced-SAM-wd-5e-05-920', verbose=True)
            
            print("❌ Wrong configuration should have failed but didn't!")
        except Exception as e:
            print(f"✅ Correctly failed with wrong configuration: {e}")
            print(f"  Error type: {type(e).__name__}")
        
        # 测试正确的配置
        print("\nTesting with CORRECT configuration (should succeed):")
        try:
            correct_model = ActorCriticICP(
                num_actor_obs=num_actor_obs,
                num_critic_obs=num_actor_obs,
                num_actions=6,
                icp_point_dim=config['icp_point_dim'],
                icp_num_points=config['num_points'],
                icp_weights_path='/home/steve/corn/ckpts/512-32-balanced-SAM-wd-5e-05-920',
                freeze_icp=True,
                actor_hidden_dims=config['actor_hidden_dims'],
                critic_hidden_dims=config['actor_hidden_dims'],
                fusion_hidden_dims=config['fusion_hidden_dims'],
                activation="elu",
                init_noise_std=1.0
            )
            print("✅ Correct configuration loaded successfully!")
            
            # 测试前向传播
            actions = correct_model.act(observations)
            values = correct_model.evaluate(observations)
            print(f"  Actions shape: {actions.shape}")
            print(f"  Values shape: {values.shape}")
            
            # 特征可视化测试
            print("\n" + "="*60)
            print("ICP Feature Visualization Test...")
            print("="*60)
            
            # 创建不同形状的点云进行测试 - 使用正确的维度
            test_cases = [
                ("Sphere", torch.randn(1, 512, 3) * 0.5),  # 球形点云
                ("Plane", torch.cat([torch.randn(1, 512, 2), torch.zeros(1, 512, 1)], dim=2)),  # 平面点云
                ("Line", torch.cat([torch.randn(1, 512, 1), torch.zeros(1, 512, 2)], dim=2)),  # 线形点云
            ]
            
            # 碰撞检测测试 - 创建明显会碰撞和不会碰撞的场景
            print("\n" + "="*60)
            print("Collision Detection Accuracy Test...")
            print("="*60)
            
            # 测试手部状态：正常手部状态
            normal_hand_state = torch.randn(1, 9) * 0.1  # 小手部状态变化
            normal_context = {'hand_state': normal_hand_state}
            
            # 测试手部状态：极端手部状态（可能碰撞）
            extreme_hand_state = torch.randn(1, 9) * 2.0  # 大手部状态变化
            extreme_context = {'hand_state': extreme_hand_state}
            
            collision_test_cases = [
                # 明显不会碰撞的场景
                ("Empty Space", torch.randn(1, 512, 3) * 10.0, normal_hand_state, "Low collision expected"),
                ("Far Away Sphere", torch.randn(1, 512, 3) * 5.0 + 10.0, normal_hand_state, "Low collision expected"),
                
                # 明显会碰撞的场景
                ("Hand-Sized Sphere", torch.randn(1, 512, 3) * 0.1, extreme_hand_state, "High collision expected"),
                ("Dense Cluster", torch.randn(1, 512, 3) * 0.05, extreme_hand_state, "High collision expected"),
                
                # 边界情况
                ("Medium Sphere", torch.randn(1, 512, 3) * 0.5, normal_hand_state, "Medium collision expected"),
                ("Large Sphere", torch.randn(1, 512, 3) * 2.0, extreme_hand_state, "Medium-High collision expected"),
            ]
            
            print("\nCollision probability predictions:")
            print("-" * 80)
            print(f"{'Scene':<20} {'Hand State':<15} {'Collision Prob':<15} {'Expected':<20}")
            print("-" * 80)
            
            for name, point_cloud, hand_state, expected in collision_test_cases:
                context = {'hand_state': hand_state.unsqueeze(0)}  # 添加batch维度
                
                with torch.no_grad():
                    icp_output, icp_features = correct_model.icp_encoder(point_cloud, context)
                
                # 计算碰撞概率（取所有patch的平均值）
                collision_prob = torch.sigmoid(icp_output['collision']).mean().item()
                
                # 判断预测是否合理
                if "Low" in expected and collision_prob < 0.3:
                    result = "✅ Correct"
                elif "High" in expected and collision_prob > 0.7:
                    result = "✅ Correct"
                elif "Medium" in expected and 0.3 <= collision_prob <= 0.7:
                    result = "✅ Correct"
                else:
                    result = "❌ Unexpected"
                
                hand_state_norm = torch.norm(hand_state).item()
                print(f"{name:<20} {hand_state_norm:<15.3f} {collision_prob:<15.3f} {expected:<20} {result}")
            
            # 统计测试结果
            print("\n" + "="*60)
            print("Collision Detection Summary:")
            print("="*60)
            print("The collision header should show:")
            print("- Higher probabilities for dense, close objects")
            print("- Lower probabilities for distant, sparse objects")
            print("- Sensitivity to hand state changes")
            print("- Reasonable probability ranges (0-1)")
            
            # 创建测试上下文
            test_hand_state = torch.randn(1, 9)  # 1个样本，9维手部状态
            test_context = {'hand_state': test_hand_state}
            
            for name, point_cloud in test_cases:
                with torch.no_grad():
                    icp_output, icp_features = correct_model.icp_encoder(point_cloud, test_context)
                    
                print(f"\n{name} point cloud:")
                print(f"  Input shape: {point_cloud.shape}")
                print(f"  ICP features shape: {icp_features.shape}")
                print(f"  Collision output shape: {icp_output['collision'].shape}")
                print(f"  Collision output mean: {icp_output['collision'].mean().item():.4f}")
                print(f"  Collision output std: {icp_output['collision'].std().item():.4f}")
                
                # 特征统计
                features_mean = icp_features.mean().item()
                features_std = icp_features.std().item()
                features_min = icp_features.min().item()
                features_max = icp_features.max().item()
                
                print(f"  Features stats - Mean: {features_mean:.4f}, Std: {features_std:.4f}")
                print(f"  Features range - Min: {features_min:.4f}, Max: {features_max:.4f}")
            
        except Exception as e:
            print(f"❌ Correct configuration failed: {e}")
            print(f"  Error type: {type(e).__name__}")
        
        # 检查权重文件
        checkpoint = torch.load('/home/steve/corn/ckpts/512-32-balanced-SAM-wd-5e-05-920', map_location='cpu')
        print("\nCheckpoint keys:", checkpoint.keys())

        # 验证权重使用情况
        print("\n" + "="*60)
        print("Verifying ICP weight usage...")
        print("="*60)
        
        # 创建测试数据
        test_point_cloud = torch.randn(2, 512, 3)  # 2个样本，512个点，3维
        test_hand_state = torch.randn(2, 9)  # 2个样本，9维手部状态
        test_context = {'hand_state': test_hand_state}
        
        # 测试ICP前向传播
        with torch.no_grad():
            icp_output, icp_features = correct_model.icp_encoder(test_point_cloud, test_context)
        
        print("✅ ICP forward pass successful!")
        print(f"  ICP output shape: {icp_output['collision'].shape}")
        print(f"  ICP features shape: {icp_features.shape}")
        print(f"  ICP output mean: {icp_output['collision'].mean().item():.4f}")
        
        # 检查权重是否被加载
        print("\nWeight loading verification:")
        
        # 检查patch_encoder权重
        patch_encoder_loaded = correct_model.icp_encoder.patch_encoder.state_dict()
        print(f"  Patch encoder weights loaded: {len(patch_encoder_loaded)} parameters")
        
        # 检查pos_embed权重
        pos_embed_loaded = correct_model.icp_encoder.pos_embed.state_dict()
        print(f"  Position embedding weights loaded: {len(pos_embed_loaded)} parameters")
        
        # 检查tokenize权重
        tokenize_loaded = correct_model.icp_encoder.tokenizes.state_dict()
        print(f"  Tokenize weights loaded: {len(tokenize_loaded)} parameters")
        
        # 检查encoder权重
        encoder_loaded = correct_model.icp_encoder.encoder.state_dict()
        print(f"  Encoder weights loaded: {len(encoder_loaded)} parameters")
        
        # 检查header权重
        header_loaded = correct_model.icp_encoder.headers.state_dict()
        print(f"  Header weights loaded: {len(header_loaded)} parameters")
        
        # 验证权重值是否与checkpoint一致
        print("\nWeight value verification:")
        checkpoint_weights = checkpoint['model_state_dict']
        
        # 检查一个具体的权重值
        checkpoint_patch_weight = checkpoint_weights['patch_encoder']['mlp.model.0.linear.weight']
        loaded_patch_weight = patch_encoder_loaded['mlp.model.0.linear.weight']
        
        weight_diff = torch.abs(checkpoint_patch_weight - loaded_patch_weight).max()
        print(f"  Patch encoder weight difference: {weight_diff:.2e}")
        if weight_diff < 1e-6:
            print("  ✅ Patch encoder weights match perfectly!")
        else:
            print("  ❌ Patch encoder weights don't match!")
        
        # 检查编码器权重
        checkpoint_encoder_weight = checkpoint_weights['encoder']['layer.0.attention.attention.attention.Wqkv.weight']
        loaded_encoder_weight = encoder_loaded['layer.0.attention.attention.attention.Wqkv.weight']
        
        encoder_diff = torch.abs(checkpoint_encoder_weight - loaded_encoder_weight).max()
        print(f"  Encoder weight difference: {encoder_diff:.2e}")
        if encoder_diff < 1e-6:
            print("  ✅ Encoder weights match perfectly!")
        else:
            print("  ❌ Encoder weights don't match!")

        # 检查编码器权重形状
        model_state_dict = checkpoint['model_state_dict']
        print("Model state dict keys:", list(model_state_dict.keys()))
        
        # 检查编码器权重
        if 'encoder' in model_state_dict:
            encoder_dict = model_state_dict['encoder']
            print("Encoder weights:")
            
            # 统计层数
            layer_numbers = set()
            for key in encoder_dict.keys():
                if 'layer.' in key:
                    layer_num = int(key.split('.')[1])
                    layer_numbers.add(layer_num)
            
            print(f"  Number of encoder layers: {len(layer_numbers)}")
            print(f"  Layer numbers: {sorted(layer_numbers)}")
            
            # 显示每层的第一个权重
            for layer_num in sorted(layer_numbers):
                layer_keys = [k for k in encoder_dict.keys() if f'layer.{layer_num}.' in k]
                if layer_keys:
                    first_key = layer_keys[0]
                    print(f"  Layer {layer_num} - {first_key}: {encoder_dict[first_key].shape}")
            
            # 显示其他重要权重
            other_keys = [k for k in encoder_dict.keys() if 'layer.' not in k]
            for key in other_keys[:5]:
                print(f"  {key}: {encoder_dict[key].shape}")

        if 'tokenize' in model_state_dict:
            tokenize_dict = model_state_dict['tokenize']
            print("Tokenize weights:")
            for key, value in list(tokenize_dict.items())[:10]:  # 只显示前10个
                print(f"  {key}: {value.shape}")
        
        # 检查patch_encoder权重
        if 'patch_encoder' in model_state_dict:
            patch_dict = model_state_dict['patch_encoder']
            print("Patch encoder weights:")
            for key, value in list(patch_dict.items())[:10]:  # 只显示前10个
                print(f"  {key}: {value.shape}")
        
        # 检查pos_embed权重
        if 'pos_embed' in model_state_dict:
            pos_dict = model_state_dict['pos_embed']
            print("Position embedding weights:")
            for key, value in list(pos_dict.items())[:10]:  # 只显示前10个
                print(f"  {key}: {value.shape}")

        if 'header' in model_state_dict:
            header_dict = model_state_dict['header']
            print("Header weights:")
            for key, value in list(header_dict.items())[:10]:  # 只显示前10个
                print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    print("Starting ActorCriticICP Tests...")
    
    # Run main test
    success = test_actor_critic_icp()
    
    if success:
        # Run additional tests
        test_with_different_inputs()
    else:
        print("Main test failed, skipping additional tests.")
    
    print("\nTest completed!")
