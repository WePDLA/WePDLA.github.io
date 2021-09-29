---
layout: post
title: TEXT2EVENT: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction
comments: True
author: 阚志刚
---

作者： 阚志刚

# 一、简介

这篇文章是来自中科院软件研究所的文章[链接](https://aclanthology.org/2021.acl-long.217.pdf)，发表在ACL2021上。这篇文章使用编码解码器的方法来完成篇章级事件抽取任务。这种做法的优点是，可以不需要token-level的精细标注，只需要record-level的粗粒度标注即可。考虑到任务的输入是一段文本，是一个序列，而输出的事件是一个结构化的信息，无法直接使用解码器进行生成。因此本文还提出了一种可逆的将结构化事件信息转换成线性表示的方法。

# 二、模型介绍

## 1、编码

使用多层transformer进行编码，其实就是使用一个预训练语言模型来编码，文章中用的是T5。表示为：

H = Encoder(x_1,...,x_{|x|})

## 2、解码

在编码器编码完成之后，解码器对输出进行生成。生成的顺序从前往后，每次生成新的token都需要用到已生成的信息。第i个token（y_i）的生成公式为：

y_i,h_{i}^d = Decoder([H;h_{1}^d,...,h_{i-1}^d],y_{i-1})

其中，h_i^d是decoder第i步的状态，H是Encoder的输出。解码的过程会有一个起始符“<bos>”，结束符为“<eos>”。解码器输出序列的条件概率为：

p(y|x)= \prod_{i}^{|y|}p(y_{i}|y_{<i})

# 三、结构化事件信息的线性表示

## 1、正常的事件信息表示

![image](/figures/2021-09-29-TEXT2EVENT Controllable Sequence-to-Structure Generation for End-to-end Event Extraction/event record format.png)






# 一、BERT训练

## DeepSpeed usage:

{% highlight yaml %}
$ deepspeed_train.py --local_rank=0 --cf bert_base.json --max_seq_length 128 --output_dir outputs --print_steps 10 --deepspeed --job_name train_test --deepspeed_config train_test.json --data_path_prefix data --use_nvidia_dataset --rewarmup --lr_offset 0.0
{% endhighlight %}

#  二、BERT代码实现参考

参考 [DeepSpeedExamples/bing_bert/pytorch_pretrained_bert/modeling.py](https://github.com/microsoft/DeepSpeedExamples/blob/20ea07a2a069696abec212e25476a9bf76aced70/bing_bert/pytorch_pretrained_bert/modeling.py)

# 二、从Transformer讲起
BERT模型中的每一个layer实际上是Transformer[<sup>1</sup>](#1)中的Encoder结构，在介绍BERT公式化之前，我们首先关注Transformer的attention机制。
## 2.1 Scaled Dot-Product Attention
{% include figure.html src="/figures/bert_formula/sdp.jpg" caption="图1：Scaled Dot-Product Attention。"%}
对于输入矩阵Q、K、V，经过Scaled Dot-Product Attention之后，其输出结果为：
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt d_k})V$$，其中$$d_k$$是矩阵K的维度。

## 2.2 Multi-Head Attention
{% include figure.html src="/figures/bert_formula/mha.jpg" caption="图2：Multi-Head Attention。"%}
Multi-Head Attention由若干个Scaled Dot-Product Attention级联而来，其公式如下：
$$MultiHead(Q,K,V)=Concat(head_1,head_2,\cdots,head_h)W^0$$，其中，$$head_i = Attention(QW^Q_i,KW^K_i,VW^V_i)$$。

# 三、BERT公式化表示
在本节中，BERT模型的输入均为x，为表示和计算的方便，在本文中先忽略掉Dropout函数。
## 3.1 BERTLayerNorm
{% include figure.html src="/figures/bert_formula/ln.png" caption="图3：BERTLayerNorm。 "%}
LayerNorm的公式如下：
$$\mu = \frac{1}{H}\sum\limits_{i=1}^H x_i$$
$$\sigma = \sqrt{\frac{1}{H}}\sum\limits_{i=1}^H (x_i-\mu)^2$$
$$y=LN(x)=g\odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + b$$，其中，$\odot$是element-wise相乘，g和b是可学习的参数。

## 3.2 BERTEmbedding
{% include figure.html src="/figures/bert_formula/embedding.png" caption="图4：BERTEmbedding。 "%}
主要关注代码中的forward函数。Embedding层的输入为x，position_embedding和token_type_embedding都是基于输入x而做的变换。
令$$f_{EM}=nn.Embedding$$，计算公式如下：
$$word\_embeddings=f_{EM1}(x)$$
$$position\_embeddings=f_{EM2}(torch.arange(x.size(1)))$$
$$token\_type\_embedding=f_{EM3}(torch.zeros_like(x))$$
$$E=f_{EM1}+f_{EM2}+f_{EM3}$$
$$E_o=LN(E)$$

## 3.3 BERTSelfAttention
{% include figure.html src="/figures/bert_formula/selfattention.png" caption="图5：BERTSelfAttention。 "%}
BERTSelfAttention就是最重要的attention机制部分。忽略掉代码过程中的矩阵转置等细节，实际上在这一部分，仅做了我们之前讲过的Scaled Dot-Product Attention。将这部分的数学函数定义为：$$f_{SelfAtt}$$。

## 3.4 BERTSelfOutput
{% include figure.html src="/figures/bert_formula/selfoutput.png" caption="图6：BERTSelfOutput。 "%}
在这一部分中，SelfOutput仅做了一个线性变换和DropOut。为表示方便，将此部分的数学函数定义为：$$f_{SelfOut}$$。

## 3.5 BERTAttention
{% include figure.html src="/figures/bert_formula/attention.png" caption="图7：BERTAttention。 "%}
BERTAttention中包括两个部分：BERTSelfAttention和BERTSelfOutput，同样的，将此部分的数学函数定义为：$$f_{Att}$$。

## 3.6 BERTIntermediate
{% include figure.html src="/figures/bert_formula/intermediate.png" caption="图8：BERTIntermediate。 "%}
BERTIntermediate模块实际上做了一个activation的操作，将此部分的数学函数定义为：$$f_{Inter}$$。

## 3.7 BERTOutput
{% include figure.html src="/figures/bert_formula/output.png" caption="图9：BERTOutput。 "%}
BERTOutput与BERTSelfOutput类似，同样是做了线性变换和Dropout，数学函数为$$f_{Out}$$。

## 3.8 BERTLayer
{% include figure.html src="/figures/bert_formula/layer.png" caption="图10：BERTLayer。 "%}
BERTLayer模块的作用是BERT模型中的具体层进行选择，在这个代码中，一个BERTLayer实际上是一个Encoder block，假设输入为x，数学函数为$$f_{E}$$，其数学表达式为：
$$f_{E}=Attention(LN(x))+x+Linear(Inter(LN(LN(x)+Attention(LN(x))+x)))$$
即：
$$f_{E}=f_{Att}(f_{LN}(x))+x+f_{Out}(f_{Inter}(f_{LN}(f_{LN}(x)+f_{Att}(LN(x))+x)))$$。
这是一个Encoder block中的公式表示。

## 3.9 BERTEncoder
{% include figure.html src="/figures/bert_formula/encoder.png" caption="图11：BERTEncoder。 "%}
该模块在BERT模型中表示的整个的Encoder模块。

## 3.10 BERTPooler
{% include figure.html src="/figures/bert_formula/pooler.png" caption="图12：BERTPooler。 "%}
在所有的Encoder输出之后进行Pooler，数学表示为$$f_{Pooler}=tanh(f_{Linear})$$。

# 四、BERT公式汇总
输入为x。
## 4.1 第一阶段Embedding
$$word\_embeddings=f_{EM1}(x)$$
$$position\_embeddings=f_{EM2}(torch.arange(x.size(1)))$$
$$token\_type\_embedding=f_{EM3}(torch.zeros_like(x))$$
$$E=f_{EM1}+f_{EM2}+f_{EM3}$$
$$E_o=f_{LN}(E)$$
## 4.2 第二阶段Encoder
$$f_{E1}=f_{Att}(f_{LN}(x))+x+f_{Out}(f_{Inter}(f_{LN}(f_{LN}(x)+f_{Att}(f_{LN}(x))+x)))$$
$$f_{E2}=f_{Att}(f_{LN}(f_{E1}))+f_{E1}+f_{Out}(f_{Inter}(f_{LN}(f_{LN}(f_{E1})+f_{Att}(f_{LN}(f_{E1}))+f_{E1})))$$
$$\cdots$$
$$f_{E12}=f_{Att}(f_{LN}(f_{E11}))+f_{E11}+f_{Out}(f_{Inter}(f_{LN}(f_{LN}(f_{E11})+f_{Att}(f_{LN}(f_{E11}))+f_{E11})))$$

## 4.3 第三阶段Pooler
$$f_{Pooler}=tanh(f_{Linear}(f_{E12}))$$。


## 参考文献
<div id="1">
- [1] [Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.] (https://arxiv.org/abs/1706.03762)
</div>



## 2.1 class BertMultiTask
```python
class BertMultiTask:
    def __init__(self, args):
        if not args.use_pretrain:
            if args.progressive_layer_drop:
                ...
            else:
                from nvidia.modelingpreln import BertForPreTrainingPreLN, BertConfig

            bert_config = BertConfig(**self.config["bert_model_config"])
            self.network = BertForPreTrainingPreLN(bert_config, args)
```


### 2.2 class BertForPreTrainingPreLN
```python
class BertForPreTrainingPreLN(BertPreTrainedModel):
    def __init__(self, config, args):
        self.bert = BertModel(config, args)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)

    def forward(self, batch, log=True):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False

        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations)

        if masked_lm_labels is not None and next_sentence_label is not None:
            # filter out all masked labels.
            masked_token_indexes = torch.nonzero(
                (masked_lm_labels + 1).view(-1)).view(-1)
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_token_indexes)
            target = torch.index_select(masked_lm_labels.view(-1), 0,
                                        masked_token_indexes)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), target)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
```



### 2.3 class BertModel
```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config, args=None):
        self.embeddings = BertEmbeddings(config)
        ...
        self.encoder = BertEncoder(
            config, args, sparse_attention_config=self.sparse_attention_config)
        self.pooler = BertPooler(config)
        ...

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True,
                checkpoint_activations=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If BertEncoder uses sparse attention, it needs to be padded based on the sparse attention block size
        if self.sparse_attention_config is not None:
            pad_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self.sparse_attention_utils.pad_to_block_size(
                block_size=self.sparse_attention_config.block,
                input_ids=input_ids,
                attention_mask=extended_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=None,
                inputs_embeds=None,
                pad_token_id=self.pad_token_id,
                model_mbeddings=self.embeddings)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        # If BertEncoder uses sparse attention, and input_ids were padded, sequence output needs to be unpadded to original length
        if self.sparse_attention_config is not None and pad_len > 0:
            encoded_layers[-1] = self.sparse_attention_utils.unpad_sequence_output(
                pad_len, encoded_layers[-1])

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
```



#### 2.3.1 class BertEmbeddings
```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```


#### 2.3.2 class BertEncoder
```python
class BertEncoder(nn.Module):
    def __init__(self, config, args, sparse_attention_config=None):
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        if args.deepspeed_transformer_kernel:
            ...
        else:
            layer = BertLayer(config)
            if sparse_attention_config is not None:
                from deepspeed.ops.sparse_attention import BertSparseSelfAttention
                layer.attention.self = BertSparseSelfAttention(
                    config, sparsity_config=sparse_attention_config)
            self.layer = nn.ModuleList([
                copy.deepcopy(layer) for _ in range(config.num_hidden_layers)
            ])
    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True,
                checkpoint_activations=False):
        all_encoder_layers = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(
                    custom(l, l + chunk_length), hidden_states,
                    attention_mask * 1)
                l += chunk_length
            # decoder layers
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

```

#### 2.3.2.1 class BertLayer
```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                   eps=1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                    eps=1e-12)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        input_layer_norm = self.PreAttentionLayerNorm(hidden_states)
        attention_output = self.attention(input_layer_norm, attention_mask)

        intermediate_input = hidden_states + attention_output

        intermediate_layer_norm = self.PostAttentionLayerNorm(
            intermediate_input)
        intermediate_output = self.intermediate(intermediate_layer_norm)
        layer_output = self.output(intermediate_output)

        return layer_output + intermediate_input
```


#### 2.3.2.1.1 class BertAttention、BertLayerNorm、BertIntermediate、BertOutput
```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias
            
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.intermediate_size,
                                          act=config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```



LayerNorm的公式如下：

$$\mu = \frac{1}{H}\sum\limits_{i=1}^H x_i$$

$$\sigma = \sqrt{\frac{1}{H}}\sum\limits_{i=1}^H (x_i-\mu)^2$$

$$
LN(x)=g\odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + b
\tag{1}\label{LN}
$$

其中，$$\odot$$是element-wise相乘，$$g$$ 和 $$b$$ 是可学习的参数。



#### 2.3.2.1.1 class BertSelfAttention、BertSelfOutput
```python
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        pdtype = attention_scores.dtype
        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```


#### Attention

{% include figure.html height="360" width="312" src="/figures/bert_formula/sdp.jpg" caption="图1：Scaled Dot-Product Attention"%}

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\tag{2}\label{Attention}
$$

$$Q\in \mathbb{R}^{n\times d_k}, K\in\mathbb{R}^{m\times d_k}, V\in\mathbb{R}^{m\times d_v}$$

$$Attention(q_t,K,V)=\sum_{s=1}^m \frac{1}{Z} exp(\frac{<q_t,k_s>}{\sqrt{d_k}})V_s$$


#### Multi-Head Attention

{% include figure.html height="360" width="312" src="/figures/bert_formula/mha.jpg" caption="图2：Multi-Head Attention"%}

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

$$W_i^Q \in \mathbb{R}^{d_k\times \tilde{d_k}}, 
W_i^K \in \mathbb{R}^{d_k\times \tilde{d_k}}, 
W_i^V \in \mathbb{R}^{d_v\times \tilde{d_v}}, $$

$$
MultiHead(Q,K,V)=Concat(head_i, \cdots, head_h)
\tag{3}\label{MHA}
$$


##### Self_Attention

$$Attention(X,X,X), MultiHead(X,X,X)$$


#### 2.3.3 class BertPooler
```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act="tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output
```
            


### 2.4 class BertPreTrainingHeads

```python
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self,
                sequence_output,
                pooled_output,
                masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output,
                                             masked_token_indexes)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
```


```python
class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states, masked_token_indexes):
        hidden_states = self.transform(hidden_states)

        if masked_token_indexes is not None:
            hidden_states = torch.index_select(
                hidden_states.view(-1, hidden_states.shape[-1]), 0,
                masked_token_indexes)

        torch.cuda.nvtx.range_push(
            "decoder input.size() = {}, weight.size() = {}".format(
                hidden_states.size(), self.decoder.weight.size()))
        hidden_states = self.decoder(hidden_states) + self.bias
        torch.cuda.nvtx.range_pop()
        return hidden_states
```


```python
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act=config.hidden_act)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
```


```python
@torch.jit.script
def f_gelu(x):
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)


@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LinearActivation(Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        if isinstance(act, str) or (sys.version_info[0] == 2
                                    and isinstance(act, unicode)):
            if bias and act == 'gelu':
                self.fused_gelu = True
            elif bias and act == 'tanh':
                self.fused_tanh = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.fused_gelu:
            return bias_gelu(self.bias, F.linear(input, self.weight, None))
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
```



LinearActivation

$$tanh(x)=tanh(y)$$

$$ gelu(x) = 0.5*y*(1.0+erf(y/1.41421)) $$

其中， 

$$y=W*x+b$$

$$erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}dt$$




# 三、BERT公式化表示


## 3.1 前向推理过程的公式化表示

### 3.1.1 BERTEmbedding

仅考虑单一句子输入时，Embedding层的输入为 $$x$$，其物理意义为input_ids。BERTEmbedding是三种 embeddings 的融合，word_embedding，position_embedding 和 token_type_embedding，这三种 embeddings 都是基于输入x而做的变换。
BERTEmbedding的表达公式为：

$$x_{EM} = f_{EM} (x) = LN(E(x))$$

其中，$$LN(\cdot)$$ 见 Eq. $$\eqref{LN}$$,

$$E(x) = words\_embeddings(x) + position\_embeddings(x) + token\_type\_embeddings(x)$$


### 3.1.2 BERTLayer

第 $$i$$ 层Bert的输入为 $$x_i$$，输出为 $$$x_{i+1}$$，有表达式如下：

$$
x_{i+1} = layer(x_i) = W* x_{io}  + x_{ia}
$$

其中，

$$x_1 = LN(x_i)$$

$$x_2 = Attention(x_1)$$

$$x_{ia} = x_i + x_2$$

$$x_3 = LN(x_{ia})$$

$$x_{io}  = gelu(x_3)$$


### 3.1.3 BERTPooler

在$$L$$层之后，对Encoder输出 $$x_{L}$$ 进行Pooler，表达式为：

$$x_{Pooler}=tanh(x_{L})$$



## 3.2 BERT训练公式化表示

目标函数为：

$$ total\_loss = masked\_lm\_loss + next\_sentence\_loss $$
