---
layout: post
mathjax: true
catalog: true
title: Autotuning of DeepSpeed的自动并行方案 
comments: True
author: liangpeng

---

2021年11月15日，DeepSpeed发布了他们的自动化获取训练策略的方案：[Autotuning](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)，
其本质是对ZeRO stage和stage相对应的ZeRO配置，以及采用梯度累计的策略下micro_batch_size大小的自动化搜索。
本文章将对其自动化搜索的过程以及源码进行分析。

总结：Autotuning本质上是超参数的搜索，并没有对数据并行、模型并行的策略进行修改。
他根据不同的超参数配置，自动生成多个实验来计算不同配置下的性能，并从中选择最优的超参数配置。
对于炼丹师们来说，还是一个可以快速找到最优性能的方法。

不足：目前Autotuning中的显存计算方法其实跟其实现的逻辑还是有一些区别的，且实测ZeRO3目前还面临着显存泄露的问题，
需要重新实现模型来规避。另外，在显存的计算上，也没有考虑到torch在cuda初始化时所产生的固定开销。

```text
注意：1. DeepSpeed团队把前向过程产生的中间结果(intermediate results
        或feature_maps或intermediate activation)叫做激活值(activation)
     2. ZeRO stages, micro-batch sizes和其他的配置也可以被用户配置覆盖。 
```

## 工作流程

1. 在autotuning的开始，Autotuner会先做一个profile的工作，来分析所需运行模型的参数量，
以及激活值的内存。
   其实就是跑一遍前向然后结束进程。以前研究DeepSpeed的运行runtime的时候，就觉得他是老套娃了。
   下面看看源码。只对profile过程感兴趣的可以直接看第五步。
   1. Autotuner.model_info_profile_run()
    
        这一个过程就是起了一个小experiment来获取参数量和激活值大小。
        ```python
        # 配置运行信息
        ds_config = copy.deepcopy(self.user_config)
        ds_config[AUTOTUNING] = {
            "enabled": True,
            "model_info_path": model_info_path,
            "model_info": {
                "profile": True
            }
        }
        
        exp_config = {}
        exp_name = "profile_model_info"
        exp_config['name'] = exp_name
        exp_config[DS_CONFIG] = ds_config # 
        exp_config['num_gpus'] = self.exp_num_gpus # 等于args.num_gpus 或者self.rm.num_gpus_per_node
        exp_config['num_nodes'] = self.exp_num_nodes # 等于args.num_nodes或者self.rm.nodes
        exp_path = os.path.join(self.exps_dir, f'{exp_name}.json')
    
        with open(exp_path, 'w', buffering=BUFSIZE) as fd:
            json.dump(exp_config, fd)
            fd.flush()
            os.fsync(fd)
        
        # self.rm: ResourceManager类
        self.rm.schedule_experiments([exp_path])
        self.rm.run() # 接着看这个
        ```
   2.  ResourceManager.run()
    
        调用self.run_job(exp, reservations)，其中reservations为可用GPU设备信息。
      
   3. ResourceManager.run_job(exp, reservations)
        
        启动线程运行run_experiment函数
        ```python
        t = threading.Thread(target=run_experiment,
                             args=(exp,
                                   reservations,
                                   user_script,
                                   user_args))
        t.start()
        ```
   4. run_experiment(exp, reservations, user_script, user_args)
    
        利用subprocess库执行cmd命令。cmd一个例子为
        ```bash
        deepspeed --force_multi --include localhost:2 --master_port 12345 my_model_train.py --ds_config_path ds_config.json
        ```
        ```python
        exp["user_script"] = user_script
        exp["user_args"] = user_args
    
        cmd = ["deepspeed"] + exp["launcher_args"] + [user_script] + user_args
    
        ...
       
       
        with open(os.path.join(exp_dir, "stdout.log"), "wb") as out, open(
            os.path.join(exp_dir, "stderr.log"), "wb"
        ) as err:
            result = subprocess.Popen(cmd, stdout=out, stderr=err)
            result.wait()
            out.flush()
            err.flush()
            os.fsync(out)
            os.fsync(err)
        ```
      
    5. 中间的启动过程略，直接看核心profile代码：
       
       1. 参数量获取：在DeepSpeedEngine初始化时计算，就是计算model.parameters()的所有tensor的numel()之和。
        ```python
        def _get_model_parameters(self):
            if self.autotuning_profile_model_info():
                self.autotuning_model_info = {}
                num_params = 0
                trainable_num_params = 0
    
                for p in self.module.parameters():
                    # since user code might call deepspeed.zero.Init() before deepspeed.initialize(), need to check the attrbuite to check if the parameter is partitioned in zero 3 already or not
                    n = 0
                    if hasattr(p, "ds_tensor"):  # if the parameter is partitioned in zero 3
                        n += p.ds_numel
                    else:  # if the parameter is not partitioned in zero 3 yet
                        n += p.numel()
                    num_params += n
                    if p.requires_grad:
                        trainable_num_params += n
                if self.global_rank == 0:
                    self.autotuning_model_info[
                        "num_params"] = num_params * self.mp_world_size
                    self.autotuning_model_info[
                        "trainable_num_params"] = trainable_num_params * self.mp_world_size
    
                print(f"model parameter = {num_params}")

        ```
       
       2. activation获取：利用torch.cuda.memory_allocated()来获取，
       其实就是前向执行前后的memory_allocated值之差。
        ```python
       if self.autotuning_profile_model_info():
            ma = get_ma_status()
        else:
            see_memory_usage("Engine before forward", force=self.memory_breakdown())
       
       ...
       
       loss = self.module(*inputs, **kwargs)
       
       ... 
       
        if self.autotuning_profile_model_info():
            activation_mem = get_ma_status() - ma
            self.autotuning_model_info["activation_mem_per_gpu"] = activation_mem
            print_json_dist(self.autotuning_model_info,
                            [0],
                            path=self.autotuning_model_info_path())
            exit()
        else:
        ```
   
2. 紧接着，Autotuner会以[0, 1, 2, 3]的顺序先搜索ZeRO的stage，估计每个GPU在训练模型时所需的最小memory，
   也就是ZeRO实例化时所需的显存量，并且与当前的GPU可用显存进行比较。如果小于GPU可用显存，则说明该stage可以运行。
   
    Memory计算的核心代码：
   
    ```python
    def get_instantiation_memory_required_per_gpu(self, zero_stage):
        num_params = self.get_model_num_params()
        total_gpus = self.exp_num_nodes * self.exp_num_gpus
        fp16_enabled = self.fp16_enabled()

        if not num_params:
            return 0
        # assume the model uses Adam optimizer
        # ZERO_OPTIMIZATION_DISABLED:
        params_mem = num_params * (2 if fp16_enabled else 4)
        gradients_mem = num_params * (2 if fp16_enabled else 4)
        optimizer_mem = num_params * (16 if fp16_enabled else 8)

        if zero_stage >= ZERO_OPTIMIZATION_OPTIMIZER_STATES:
            optimizer_mem = optimizer_mem / total_gpus

        if zero_stage >= ZERO_OPTIMIZATION_GRADIENTS:
            gradients_mem = gradients_mem / total_gpus

        if zero_stage >= ZERO_OPTIMIZATION_WEIGHTS:
            params_mem = params_mem / total_gpus

        mem_per_gpu = (params_mem + gradients_mem + optimizer_mem) / self.mp_size()

        return mem_per_gpu
    ```
   
    紧接着Autotuner会尝试搜索在该stage下每一个GPU的micro-batch的大小，以及其他的ZeRO设置：
    1. Autotuner会先选择一系列可行的micro-batch大小（用户可以指定模型训练中的最大全局训练batch大小），
    然后接着搜索。
    
        micro-batch-size空间的计算规则非常简单，利用GPU总显存减去ZERO实例化所需显存后，除以激活值的memory，
       则1到max_micro_batch_size都是待选项，DeepSpeed默认会遍历所有可能的micro_batch_size，
       也可用户设置需要遍历的micro_batch_size的数量。
       ```python
       calculated_max_micro_batch_size = int(
            self.gpu_mem - 
            self.get_instantiation_memory_required_per_gpu(stage)) // self.activation_mem)
        ```

    2. 每一个ZeRO stage都有自己的一套搜索空间来决定其他的ZeRO配置。用户也可以通过json文件覆盖。
    其逻辑在autotuner.py中的Autotuner._generate_experiments方法中实现。
       
    3. Autotuner会使用像xgboost model-based algorithm一类的算法对不同micro-batch大小和ZeRO配置的组合进行实验。 
       用户可以通过配置启发式方法来实现策略搜索的提前终止。
       ```python
        exps = self._generate_experiments(tuning_space, max_train_batch_size_per_gpu)

        logger.info(f'Tuner type is {self.autotuning_config.tuner_type}')
        if self.autotuning_config.tuner_type == AUTOTUNING_TUNER_MODELBASED:
            t = ModelBasedTuner(exps, self.rm, self.metric(), tuning_space)
        elif self.autotuning_config.tuner_type == AUTOTUNING_TUNER_RANDOM:
            t = RandomTuner(exps, self.rm, self.metric())
        else:
            t = GridSearchTuner(exps, self.rm, self.metric())

        sample_size = len(self.rm.nodes) * self.rm.num_gpus_per_node // (
            self.exp_num_gpus * self.exp_num_nodes)
        num_exps = t.tune(sample_size=sample_size,
                          n_trials=self.autotuning_config.tuner_num_trials,
                          early_stopping=self.autotuning_config.tuner_early_stopping)
        exp = t.best_exp
        metric_val = t.best_metric_val
        ```
    4. 得到基于吞吐率（或延时、FLOPs等指标）的最优配置。
        ```python
        full_best_record = self.get_best_space_record(tuning_space_name)
        full_best_metric_val = full_best_record[1] if full_best_record else -1

        if full_best_metric_val > fast_best_metric_val:
            best_metric_val = full_best_metric_val
            best_mbs = full_best_record[0][DS_CONFIG][
                TRAIN_MICRO_BATCH_SIZE_PER_GPU] if full_best_record else -1
        else:
            best_metric_val = fast_best_metric_val
            best_mbs = fast_best_mbs

        logger.info(f"End tuning for space: {tuning_space_name}")
        return max_micro_batch_size, best_mbs, best_metric_val
        ```
    
3. 如果当前ZeRO stage的最优设置性能亚于之前其他ZeRO stage的方法，则之后其他Stage的搜索会终止。
   （因为是按顺序搜索的，默认情况下，前面的stage的最优策略应该batch-size会更小）
4. 最后，全局最优设置会通过log的文件的形式告知用户。如果--autotung设置为run，还会直接开始训练。

