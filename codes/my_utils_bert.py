import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import math

from transformers.models.bert.modeling_bert import BertPreTrainedModel,\
     BertSelfOutput, BertIntermediate, BertOutput, BertPooler, BertEncoder,\
     prune_linear_layer
BertLayerNorm = torch.nn.LayerNorm

class BertGraphSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.with_graph = config.with_graph
        self.max_seq_length = config.max_seq_length
        if self.with_graph:
            self.layernorm_key = config.layernorm_key
            self.layernorm_value = config.layernorm_value
            self.div_sqrt = config.div_sqrt
            self.gate_key_type = config.gate_key_type
            self.gate_key_layer = self.get_gate_key_layer()
            self.gate_value_type = config.gate_value_type
            self.gate_value_layer = self.get_gate_value_layer()
            self.batchnorm_key = config.batchnorm_key
            if self.batchnorm_key:
                self.batchnorm_key_layer = nn.BatchNorm2d(self.max_seq_length)
            if self.layernorm_key:
                self.LayerNormKeys = BertLayerNorm(self.attention_head_size, eps=config.layer_norm_eps)
            if self.layernorm_value:
                self.LayerNormValues = BertLayerNorm(self.attention_head_size, eps=config.layer_norm_eps)

            self.batchnorm_value = config.batchnorm_value
            if self.batchnorm_value:
                self.batchnorm_value_layer = nn.BatchNorm2d(self.max_seq_length)

            if config.input_labeled_graph:
                if config.diff_pad_zero:
                    self.num_dp = 2*config.dependency_labels_size + 2
                else:
                    self.num_dp = 2*config.dependency_labels_size + 1
            else:
                if config.diff_pad_zero:
                    self.num_dp = 5
                else:
                    self.num_dp = 4#    here

            if config.diff_pad_zero:
                self.dp_relation_k = nn.Embedding(self.num_dp, self.attention_head_size, padding_idx=self.num_dp-1)
                self.dp_relation_v = nn.Embedding(self.num_dp, self.attention_head_size, padding_idx=self.num_dp-1)
            else:
                self.dp_relation_k = nn.Embedding(self.num_dp, self.attention_head_size)
                self.dp_relation_v = nn.Embedding(self.num_dp, self.attention_head_size)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def get_gate_key_layer(self):
        if self.gate_key_type == "key":
            gate_key_layer = nn.Sequential(
                nn.Linear(self.attention_head_size, self.num_attention_heads),
                nn.Sigmoid())
            return gate_key_layer
        else:
            return None

    def get_gate_key(self, dp_keys):
        if self.gate_key_type == "key":
            gate_values = self.gate_key_layer(dp_keys).transpose(1, 3)
        return gate_values


    def get_gate_value_layer(self):
        if self.gate_value_type == "value":
            gate_value_layer = nn.Sequential(
                nn.Linear(self.max_seq_length, self.num_attention_heads),
                nn.Sigmoid())
            return gate_value_layer
            # context_layer [8, 12, 128, 64]
            # dp_values [8, 128, 128, 64]
        else:
            return None


    def get_gate_value(self, dp_values):
        if self.gate_value_type == "value":
            dp_values = dp_values.permute(0, 1, 3, 2)
            gate_values = self.gate_value_layer(dp_values)
            gate_values = gate_values.permute(0, 3, 1, 2)
        return gate_values

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    
    def relative_matmul_dp(self,x,z):
        """ Helper function for dependency parsing relations"""
        x_t = x.transpose(1,2)
        z_t = z.transpose(2,3)
    
        out = torch.matmul(x_t,z_t)
    
        out = out.transpose(1,2)
    
        return out

    def relative_matmul_dpv(self,x,z):
        """ Helper function for dependency parsing relations"""
        
        x = x.transpose(1,2)
        out = torch.matmul(x,z)
        out = out.transpose(1,2)
    
        return out
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        graph=None
    ):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Adds dependency relationships to attention weights
        if self.with_graph:
            dp_keys = self.dp_relation_k(graph.to(key_layer.device))
            if self.layernorm_key:
                dp_keys = self.LayerNormKeys(dp_keys)
         
            if self.batchnorm_key:
                dp_keys = self.batchnorm_key_layer(dp_keys)

            dp_values = self.dp_relation_v(graph.to(key_layer.device))
            if self.layernorm_value:
                dp_values = self.LayerNormValues(dp_values)
            if self.batchnorm_value:
                dp_values = self.batchnorm_value_layer(dp_values)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        
        if self.with_graph:
            if self.gate_key_type is not None:
                attention_scores = attention_scores + self.get_gate_key(dp_keys)*self.relative_matmul_dp(query_layer, dp_keys)
            else:
                if self.div_sqrt:
                    attention_scores = attention_scores + self.relative_matmul_dp(query_layer, dp_keys)/math.sqrt(self.attention_head_size)
                else:
                    attention_scores = attention_scores + self.relative_matmul_dp(query_layer, dp_keys)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        if self.with_graph:
            if self.gate_value_type is not None:
                context_layer = context_layer + self.get_gate_value(dp_values)*self.relative_matmul_dpv(attention_probs, dp_values)
            else:
                context_layer = context_layer + self.relative_matmul_dpv(attention_probs, dp_values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertGraphAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertGraphSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        graph=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, graph
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertGraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertGraphAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertGraphAttention(config)#
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        graph=None
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,graph)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, graph, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertGraphEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.with_graph = config.with_graph
        if self.with_graph:
           self.input_labeled_graph = config.input_labeled_graph
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, graph_labels=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        ## adding dependency label embeddings to others
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        print(embeddings.shape)
        # if self.with_graph:
        #     label_embeddings = self.word_embeddings(graph_labels[:,:,0])
        #     embeddings = embeddings + label_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class BertGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertGraphLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        graph=None
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, graph
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertGraphModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertGraphEmbeddings(config)
        self.encoder = BertGraphEncoder(config)
        self.pooler = BertPooler(config)
        self.with_graph = config.with_graph
        if self.with_graph:
            self.input_labeled_graph = config.input_labeled_graph
            if self.input_labeled_graph:
               self.dependency_labels_size = config.dependency_labels_size
               self.bias_dependency_label = config.bias_dependency_label
               self.unk_label_id = config.unk_label_id
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _build_graph(self, indices_batched):
        """
        Given the input graph indices of the size batch_size*max_length, creates the graph of size
        max_length*max_length, in which the upper triangle are existing edge connected with label 2 and in the lower
        triangle the existing edges are connected with label=1.
        :param indices_batched: Input graph of size batch_size*max_length, which specifies the heads for each node.
        :return: constructed graph of size batch_size*max_length*max_length.
        """

        batch_size = indices_batched.size(0)
        max_length = indices_batched.size(1)
        graph_matrix_batched = torch.zeros((batch_size, max_length, max_length))
        
        

        for counter, indices in enumerate(indices_batched):
            # TODO: speed up
            for i in range(indices.shape[0]):
                # If we truncate a sentence, and remove some of the tokens, but among them
                # there are tokens which are connected to the removed tokens in the parsing
                # this is a bug and we filter those toknes out.

                heads = indices[i]

                if heads[0].item()!= -1:
                    for head in heads:
                        if head.item() < max_length and head.item() != -100 and i != head.item() and  head.item() > -1500 :
                            graph_matrix_batched[counter,i, head.item()] = 1
            graph_matrix_batched[counter,:,:] = graph_matrix_batched[counter,:,:].transpose(0,1)*2 + graph_matrix_batched[counter,:,:]
            #assert not len((graph_matrix_batched[counter,:,:] == 3).nonzero())
            if self.config.diff_pad_zero:
                mask = (indices != -1).long()[:,0]
                mask = (mask.unsqueeze(0) * mask.unsqueeze(1)).bool()
                graph_matrix_batched[counter,:,:][~mask] = 4
                
        return graph_matrix_batched

    def _build_graph_label(self, indices_batched, labels_batched, num_dep_labels):
        """
        Given the input graph indices of the size batch_size*max_length, and  dependency labels of the size
        batch_size*max_length creates the graph of dependency labels
        :param indices_batched: Input graph of size batch_size*max_length, which specifies the heads for each node.
        :return: constructed graph of size batch_size*max_length*max_length.
        """
        batch_size = indices_batched.size(0)
        max_length = indices_batched.size(1)
        graph_matrix_batched = torch.zeros((batch_size, max_length, max_length))
        
        for counter, (indices, labels) in enumerate(zip(indices_batched, labels_batched)):
            # TODO: speed up
            for i in range(indices.shape[0]):
                
                heads = indices[i]
                label = labels[i]
                if heads[0].item() != -1:
                    for k, head in enumerate(heads):
                        if head.item() < max_length and head.item() != -100 and i != head.item() and  head.item() > -1500 :
                            if label[k].item() == 100: ## id for ['UNK'], convert to ['<l>:UNK']
                                label[k] = self.unk_label_id
                            graph_matrix_batched[counter,i, head.item()] = label[k].item() - self.bias_dependency_label + 1
            mask_matrix = (graph_matrix_batched[counter,:,:] > 0).float()
            graph_matrix_t = graph_matrix_batched[counter,:,:].transpose(0,1) + mask_matrix.transpose(0,1) * num_dep_labels
            graph_matrix_batched[counter,:,:] = graph_matrix_t + graph_matrix_batched[counter,:,:]
            mask2_matrix = graph_matrix_batched[counter,:,:] > 2* num_dep_labels
            graph_matrix_batched[counter,:,:] = graph_matrix_batched[counter,:,:] - graph_matrix_batched[counter,:,:]*mask2_matrix
        return graph_matrix_batched
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        graph_labels=None, graph=None
    ):
        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
        if self.with_graph:
            if self.input_labeled_graph:
                graph = self._build_graph_label(graph, graph_labels, self.dependency_labels_size).long()
            else:
                graph = self._build_graph(graph).long()

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long
                )  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, 
            inputs_embeds=inputs_embeds, graph_labels=graph_labels
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            graph=graph
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertGraphForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertGraphModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        graph_labels=None, graph=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            graph_labels=graph_labels,
            graph=graph
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
