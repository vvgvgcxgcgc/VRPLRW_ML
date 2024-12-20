import torch
import torch.nn as nn
import torch.nn.functional as F
# from tutel import moe as tutel_moe
from .MOELayer import MoE
 
__all__ = ['MOEModel']
 
 
class MOEModel(nn.Module):
    """
        MOE implementations:
            (1) with tutel, ref to "https://github.com/microsoft/tutel"
            (2) with "https://github.com/davidmrau/mixture-of-experts"
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']
        self.aux_loss = 0
 
        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        self.encoded_nodes = None  # shape: (batch, problem+1, EMBEDDING_DIM)
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']
 
    def pre_forward(self, reset_state):
        # depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        service_time = reset_state.node_service_time
        # node_tw_start = reset_state.node_tw_start
        # node_tw_end = reset_state.node_tw_end
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        # prob_emb = reset_state.prob_emb
        # shape: (1, 5) - only for problem-level routing/gating
 
        self.encoded_nodes, moe_loss = self.encoder( node_xy_demand)
        # print(">>>>> encoded_nodes: ", self.encoded_nodes.shape)
        # contains_nan = torch.isnan(self.encoded_nodes).any()
        # contains_greater_than_one = (self.encoded_nodes > 1).any()
        # print("Tensor contains NaN:", contains_nan)
        # print("Tensor contains value greater than 1:", contains_greater_than_one)
        # print()
        # print(self.encoded_nodes)
        self.aux_loss = moe_loss
        # shape: (batch, problem, embedding)
 
    def set_eval_type(self, eval_type):
        self.eval_type = eval_type
 
    def forward(self, state, is_greedy=False, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
 
 
        # encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        encoded_routes = get_routes_encoding(self.encoded_nodes, state.current_routes, state.cur_travel_time_routes, state.truck_num) # encoding route

        # print("___mask: ", state.mask[0])
        # print("___truck_num: ", state.truck_num[0])
        
        # print("encoded_routes: ", encoded_routes.shape)
        # sum_encoded = torch.sum(encoded_routes, dim=-1)
        # count_zero = torch.sum((sum_encoded == 0.0), dim=-1)
        # print("___count_zero: ", count_zero)
        # contains_nan = torch.isnan(encoded_routes).any()
        # contains_greater_than_one = (encoded_routes > 1).any()
        # print("Tensor contains NaN:", contains_nan)
        # print("Tensor contains value greater than 1:", contains_greater_than_one)
        # shape: (batch, pomo, embedding)
        # attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None], state.open[:, :, None]), dim=2)
        # shape: (batch, pomo, 4)
        # probs, moe_loss = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask)
        self.decoder.set_q_last(self.encoded_nodes.unsqueeze(1).expand(-1, pomo_size, -1, -1 ))
        # print("MASK:", state.mask)
 
        probs, moe_loss = self.decoder(encoded_routes, state.mask,  state.route_mask)
        # print( ' probs0: ', probs[0])
        # print( ' probs1: ', probs[1])

        self.aux_loss = self.aux_loss + moe_loss
        # shape: (batch, pomo, problem+1)
        if selected is None:
            while True:
                    if (self.training or self.eval_type == 'softmax') and not is_greedy:
                        try:
                            # print(">>>>>> probs: ", probs.shape)
                            
                            # contains_nan = torch.isnan(probs).any()
                            # contains_greater_than_one = (probs > 1).any()
                            # print("Tensor contains NaN:", contains_nan)
                            # print("Tensor contains value greater than 1:", contains_greater_than_one)
                            
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
        else:
            selected = selected
            prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
 
        return selected, prob
 
 
def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
 
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
 
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)
 
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)
 
    return picked_nodes
 
def get_routes_encoding(encoded_nodes, route_index, postion_encodings, truck_num):
    # encoded_nodes.shape: (batch, problem, embedding)
    # route_index.shape: (batch, pomo, problem,problem)
    # postion_encodings.shape: (batch, problem, 1)
    # postioned_encoded_nodes = torch.cat([encoded_nodes, postion_encodings], dim = -1)
    batch_size = route_index.size(0)
    pomo = route_index.size(1)
    n = route_index.size(2)
    embedding = encoded_nodes.size(2)

    # print('embedding 2: ',encoded_nodes[0,0])
    # zeros_tensor = torch.zeros(batch_size, 1, embedding)
    # postioned_encoded_nodes = torch.cat([postioned_encoded_nodes, zeros_tensor], dim = 1) #shape (batch, problem +1, embedding + 1 )

 
    # Mở rộng encoded_nodes để thêm chiều "pomo"
    encoded_nodes_expanded = encoded_nodes[:, None, None, :, :].expand(-1, pomo, n, -1, -1) #.unsqueeze(1).unsqueeze(2)  # Shape: (batch, 1, problem, embedding)
 
    # Sử dụng route_index để gather
    # Chuyển route_index về kích thước tương ứng với encoded_nodes_expanded
    route_index_expanded = route_index[:, :, :, :, None].expand(-1, -1, -1, -1, embedding).clone()  # Shape: (batch, pomo, problem, problem, embedding)
 
    # Gather encoded_nodes theo route_index 
    encode_routes = encoded_nodes_expanded.gather(3, route_index_expanded)  # Shape: (batch, pomo, problem, problem, embedding)
    
    encode_routes = torch.cat((encode_routes, postion_encodings[:, :, :, :-1][:, :, :, :, None]), dim = -1 )
    range_tensor = torch.arange(n)[None, None, None, :]
   
    mask = range_tensor < truck_num.unsqueeze(-1)
    mask = mask[:, :, :, :, None].expand(-1, -1, -1, -1, embedding + 1)
    encode_routes = encode_routes*mask
    # print('encode_routes0', encode_routes[0,0,0, 0])
    # print('encode_routes1', encode_routes[0,0,1, 0])
    
    # Nếu chỉ muốn shape (batch, pomo, problem, problem), bạn có thể chọn trung bình hoặc giá trị cụ thể theo embedding
    encode_routes = encode_routes.sum(3)  # Hoặc torch.mean(encode_route, dim=-1)
    node_num = torch.where(truck_num == 0, 1, truck_num)
    encode_routes = encode_routes/node_num.unsqueeze(-1)
    # print('encode_routes0', encode_routes[0,0,0])
    # print('encode_routes1', encode_routes[0,0,1])
    

 
    return encode_routes # shape(batch, pomo, problem, embedding + 1)
 
 
 
########################################
# ENCODER
########################################
 
class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        hidden_dim = self.model_params['ff_hidden_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
 
        # [Option 1]: Use MoEs in Raw Features
        if self.model_params['num_experts'] > 1 and "Raw" in self.model_params['expert_loc']:
            self.embedding_depot = MoE(input_size=2, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                       k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                       routing_method=self.model_params['routing_method'], moe_model="Linear")
            self.embedding_node = MoE(input_size=3, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                      k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                      routing_method=self.model_params['routing_method'], moe_model="Linear")
        else:
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(i, **model_params) for i in range(encoder_layer_num)])
 
    def forward(self, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 5)
        # prob_emb: (1, embedding)
 
        moe_loss = 0
        if isinstance(self.embedding_depot, MoE) or isinstance(self.embedding_node, MoE):
            # embedded_depot, loss_depot = self.embedding_depot(depot_xy)
            embedded_node, loss_node = self.embedding_node(node_xy_demand_tw)
            moe_loss = moe_loss + loss_node
        else:
            # embedded_depot = self.embedding_depot(depot_xy)
            # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand_tw)
            # shape: (batch, problem, embedding)
 
        # out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)
 
        for layer in self.layers:
            embedded_node, loss = layer(embedded_node)
            moe_loss = moe_loss + loss
 
        return embedded_node, moe_loss
        # shape: (batch, problem, embedding)
 
 
class EncoderLayer(nn.Module):
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
 
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
 
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        # [Option 2]: Use MoEs in Encoder
        if self.model_params['num_experts'] > 1 and "Enc{}".format(depth) in self.model_params['expert_loc']:
            # TODO: enabling parallelism
            # (1) MOE with tutel, ref to "https://github.com/microsoft/tutel"
            """
            assert self.model_params['routing_level'] == "node", "Tutel only supports node-level routing!"
            self.feedForward = tutel_moe.moe_layer(
                gate_type={'type': 'top', 'k': self.model_params['topk']},
                model_dim=embedding_dim,
                experts={'type': 'ffn', 'count_per_node': self.model_params['num_experts'],
                         'hidden_size_per_expert': self.model_params['ff_hidden_dim'],
                         'activation_fn': lambda x: F.relu(x)},
            )
            """
            # (2) MOE with "https://github.com/davidmrau/mixture-of-experts"
            self.feedForward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], routing_method=self.model_params['routing_method'], moe_model="MLP")
        else:
            self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
 
    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num, moe_loss = self.model_params['head_num'], 0
 
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
 
        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2, moe_loss = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2, moe_loss = self.feedForward(out2)
            out3 = input2 + out2
 
        return out3, moe_loss
 
 
########################################
# DECODER
########################################
 
class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
 
        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.final = nn.Linear(embedding_dim + 1, 1, bias=False)

 
        # [Option 3]: Use MoEs in Decoder
        if self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']:
            self.multi_head_combine = MoE(input_size=head_num * qkv_dim, output_size=embedding_dim + 1, num_experts=self.model_params['num_experts'],
                                          k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                          routing_method=self.model_params['routing_method'], moe_model="Linear")
        else:
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim + 1)
 
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention
 
    def set_kv(self, encoded_routes):
        # encoded_routes.shape: (batch, nr, embedding)
        head_num = self.model_params['head_num']
 
        self.k = reshape_by_heads1(self.Wk(encoded_routes), head_num=head_num)
        self.v = reshape_by_heads1(self.Wv(encoded_routes), head_num=head_num)
        # shape: (batch, head_num, nr, qkv_dim)
        # self.single_head_key = encoded_routes.transpose(2, 3)
        # shape: (batch,pomo, embedding + 1, nr)
 
    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads1(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)
 
    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads1(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)
    def set_q_last(self, encoded_nodes):
        head_num = self.model_params['head_num']
 
        self.q_last = reshape_by_heads1(self.Wq_last(encoded_nodes), head_num=head_num)
 
 
 
    def forward(self, encoded_routes, ninf_mask, route_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # attr.shape: (batch, pomo, 4)
        # ninf_mask.shape: (batch, pomo, problem)
 
        head_num, moe_loss = self.model_params['head_num'], 0
        self.set_kv(encoded_routes)
        #  Multi-Head Attention
        #######################################################
        # input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # # shape = (batch, group, EMBEDDING_DIM + 4)
 
        # q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # # shape: (batch, head_num, pomo, qkv_dim)
 
        # # q = self.q1 + self.q2 + q_last
        # # # shape: (batch, head_num, pomo, qkv_dim)
        # q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
 
        out_concat = multi_head_attention1(self.q_last, self.k, self.v,ninf_mask,  route_mask)
        # shape: (batch, pomo, n,  head_num*qkv_dim)
 
        if isinstance(self.multi_head_combine, MoE):
            mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
        else:
            mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo,n,  embedding)
 
        #  Single-Head Attention, for probability calculation
        #######################################################
        score = mh_atten_out
        # shape: (batch, pomo,n, problem)
 
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
 
        score_scaled = self.final(score)
        #/ sqrt_embedding_dim
        # shape: (batch, pomo, problem, nr)
 
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_clipped = score_clipped.squeeze(-1)
 
        score_masked = score_clipped + ninf_mask
        # print('score_masked', torch.mean(score_clipped, dim = -1)[0])

        # print('score_masked', score_masked[0,0])
        # print('score_masked1', score_masked[0,0])

 
        probs = F.softmax(score_masked, dim=2)
        # print('score0', score[0][0][0])
        # print('score1', score[0][0][1])
        # shape: (batch, pomo, problem)
 
        return probs, moe_loss
 
 
########################################
# NN SUB CLASS / FUNCTIONS
########################################
 
def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
 
    batch_s = qkv.size(0)
    n = qkv.size(1)
 
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)
 
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)
 
    return q_transposed
def reshape_by_heads1(qkv, head_num):
    batch_s = qkv.size(0)
    pomo = qkv.size(1)
    n = qkv.size(2)
    q_reshaped = qkv.reshape(batch_s,pomo, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)
 
    q_transposed = q_reshaped.transpose(2, 3)
    # shape: (batch, pomo head_num, n, key_dim)
 
    return q_transposed
 
 
 
def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
 
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
 
    input_s = k.size(2)
 
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)
 
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
 
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)
 
    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)
 
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
 
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
 
    return out_concat
 
def multi_head_attention1(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch,pomo, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch,pomo, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    # print()
    # print("k: ", k.shape)
    # print("q: ", q.shape)
    # print("v: ", v.shape)
 
    batch_s = q.size(0)
    pomo = q.size(1)
    head_num = q.size(2)
    n = q.size(3)
    key_dim = q.size(4)
 
    input_s = k.size(3)
 
    score = torch.matmul(q, k.transpose(3, 4))
    

    # shape: (batch, pomo, head_num, n, problem)
 
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        pass
        # score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, :,None, :, :].expand(batch_s,pomo, head_num, n, input_s)
    
    # print("route mask: ", rank3_ninf_mask[:, :,None, :, :].expand(batch_s,pomo, head_num, n, input_s)[3,0,0,30])
    # print("score0: ", score_scaled[3,0,0,30])
    # print("score1: ", score_scaled[3,0,0,32])
    weights = nn.Softmax(dim=4)(score_scaled)
    
    # print("w: ", weights[0])
    # print()
    # shape: (batch, head_num, n, problem)
    
    mask = torch.where( rank2_ninf_mask == 0.0, 1.0, 0.0)
    mask = mask[:,:, None, :,  None]
    weights = weights*mask
    # print("weight0: ", weights[3,0,0,30])
    # print("weight1: ", weights[3,0,0,32])
    out = torch.matmul(weights, v)
    # print("out: ", out[3,0,0,30])
    # print("out: ", out[3,0,0,31])
    # shape: (batch, head_num, n, key_dim)
 
    out_transposed = out.transpose(2, 3)
    # print(">>>>>>>> out_transposed: " , out_transposed.shape)
    # shape: (batch, n, head_num, key_dim)
    
    out_concat = out_transposed.reshape(batch_s, pomo, n, head_num * key_dim)
    # print(">>>>>>>> out_concat: " , out_concat.shape)
    # shape: (batch,pomo, n, head_num*key_dim)

    # assert False
    
    return out_concat
 
class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None
 
    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2
 
        return back_trans
 
 
class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
 
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
 
    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
 
        return self.W2(F.relu(self.W1(input1))), 0