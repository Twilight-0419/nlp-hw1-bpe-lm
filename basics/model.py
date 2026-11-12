import torch
import torch.nn as nn
import math
from typing import Optional,Tuple



class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super(Linear,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight = nn.Parameter(
            torch.empty(out_features,in_features, device=device, dtype=dtype)
        )
        std=math.sqrt(2.0/(in_features+ out_features))
        nn.init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3*std, b=3*std
        )

        if bias:
            self.bias=nn.Parameter(
                torch.empty(
                    out_features, device=device, dtype=dtype
                )
            )
            nn.init.zeros_(self.bias)
        else:
            self.bias=None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y=x @ self.weight.T
        if self.bias is not None:
            y+=self.bias
        return y



class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super(Embedding,self).__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.embedding_weights=nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype
                )
        )
        nn.init.trunc_normal_(
            self.embedding_weights,mean=0.0,std=1.0,a=-3.0,b=3.0
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        y=self.embedding_weights[token_ids]
        return y



class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(RMSNorm,self).__init__()
        self.d_model=d_model
        self.eps=eps
        self.gain=nn.Parameter(
            torch.empty(
                d_model,device=device,dtype=dtype
            )
        )
        nn.init.ones_(
            self.gain
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype=x.dtype
        x=x.float()
        rms=torch.sqrt(
            x.pow(2).mean(dim=-1,keepdim=True)+self.eps
        )
        y=x*self.gain/rms
        y=y.to(x_dtype)
        return y



def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super(SwiGLU,self).__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.w1=nn.Parameter(
            torch.empty(
                d_ff,d_model,device=device,dtype=dtype
            )
        )
        self.w2=nn.Parameter(
            torch.empty(
                d_model,d_ff,device=device,dtype=dtype
            )
        )
        self.w3=nn.Parameter(
            torch.empty(
                d_ff,d_model,device=device,dtype=dtype
            )
        )
        std = math.sqrt(2.0 / (d_model + d_ff))
        for w in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1=x @ self.w1.T
        y1=silu(y1)
        y2=x @ self.w3.T
        y3=y1 * y2
        return y3 @ self.w2.T



class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super(RoPE,self).__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        pos = token_positions.to(device=x.device).to(torch.float32)
        inv_freq = self.theta ** (
            -torch.arange(0, self.d_k, 2, device=x.device, dtype=torch.float32) / self.d_k
        )
        angles = pos.unsqueeze(-1) * inv_freq          # (..., seq_len, d_k/2)
        cos = angles.cos().to(x.dtype)
        sin = angles.sin().to(x.dtype)

        x_pair = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
        )
        return rotated.reshape_as(x)



def softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    max_x=torch.max(x , dim , keepdim=True).values
    standard_x=x-max_x
    logits=torch.exp(standard_x)
    return logits/torch.sum(logits,dim , keepdim=True)



def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = softmax(logits / temperature, dim=-1)
    if top_p is None or top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)


@torch.no_grad()
def generate_tokens(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    end_token_id: Optional[int] = None,
    context_length: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    single = input_ids.dim() == 1
    if single:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device=device, dtype=torch.long)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        x = input_ids if context_length is None else input_ids[:, -context_length:]
        logits = model(x)[:, -1, :]
        next_token = sample_next_token(logits, temperature, top_p)
        if end_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, end_token_id),
                next_token,
            )
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if end_token_id is not None:
            finished |= next_token.squeeze(1) == end_token_id
            if finished.all():
                break
    if single:
        input_ids = input_ids.squeeze(0)
    return input_ids


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # query: (batch_size, ..., seq_len_q, d_k)
    # key:   (batch_size, ..., seq_len_k, d_k)
    # value: (batch_size, ..., seq_len_k, d_v)
    logits = query @ torch.transpose(key,-1,-2) / (query.shape[-1])**0.5
    if mask!=None:
        logits=logits.masked_fill(~mask , float('-inf'))
    probilities = softmax(logits,dim=-1)
    return probilities @ value



class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ):
        super(CausalMultiheadSelfAttention,self).__init__()
        assert d_model % num_heads == 0
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads
        self.device=device
        self.use_rope=use_rope
        self.theta=theta
        self.max_seq_len=max_seq_len
        if use_rope:
            self.rope=RoPE(theta=theta,d_k=self.d_head,max_seq_len=max_seq_len,device=device)
        self.w_q=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.w_k=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.w_v=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.w_o=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        std = math.sqrt(1.0 / d_model)
        for w in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len=x.shape[-2]

        def split_heads(weight):
            # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_head)
            out=x @ weight.T
            out=out.reshape(*x.shape[:-1], self.num_heads, self.d_head)
            return out.transpose(-3,-2)

        query=split_heads(self.w_q)
        key=split_heads(self.w_k)
        value=split_heads(self.w_v)

        if self.use_rope:
            if token_positions is None:
                token_positions=torch.arange(seq_len,device=x.device)
            query=self.rope(query,token_positions)
            key=self.rope(key,token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len,device=x.device, dtype=torch.bool))
        out=scaled_dot_product_attention(query,key,value,mask)
        out=out.transpose(-3,-2).reshape(*x.shape)
        return out @ self.w_o.T



class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ):
        super(TransformerBlock,self).__init__()
        self.d_model=d_model
        self.numheads=num_heads
        self.d_ff=d_ff
        self.device=device
        self.dtype=dtype
        self.use_rope=use_rope
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.causalmultiheadselfattention=CausalMultiheadSelfAttention(
            d_model,
            num_heads,
            device,
            dtype,
            use_rope,
            theta,
            max_seq_len
        )
        self.swiglu=SwiGLU(
            d_model,
            d_ff,
            device,
            dtype
        )
        self.rmsnorm1=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.rmsnorm2=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y=self.rmsnorm1(x)
        z=self.causalmultiheadselfattention(y)+x
        k=self.rmsnorm2(z)
        return self.swiglu(k)+z



class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
        use_rope: bool = False,
        theta: Optional[float] = None,
    ):
        super(TransformerLM,self).__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.embedding=Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
    )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
                use_rope=use_rope,
                theta=theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])
        self.rmsnorm=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.linear=Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        x=self.embedding(input_ids)
        for layer in self.layers:
            x=layer(x)
        h=self.rmsnorm(x)
        logits=self.linear(h)
        return logits

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        end_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        return generate_tokens(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            end_token_id=end_token_id,
            context_length=self.context_length,
        )



class LSTMCell(nn.Module):
    def __init__(
        self,
        d_model: int,
        device=None,
        dtype=None,
    ):
        super(LSTMCell,self).__init__()
        self.d_model=d_model
        self.w_f=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.w_i=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.w_o=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.w_c=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.u_f=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.u_i=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.u_o=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.u_c=nn.Parameter(
            torch.empty(
                d_model,d_model,device=device,dtype=dtype
            )
        )
        self.b_f=nn.Parameter(
            torch.empty(
                d_model,device=device,dtype=dtype
            )
        )
        self.b_i=nn.Parameter(
            torch.empty(
                d_model,device=device,dtype=dtype
            )
        )
        self.b_o=nn.Parameter(
            torch.empty(
                d_model,device=device,dtype=dtype
            )
        )
        self.b_c=nn.Parameter(
            torch.empty(
                d_model,device=device,dtype=dtype
            )
        )
        std=math.sqrt(2.0/(d_model+d_model))
        for w in (self.w_f,self.w_i,self.w_o,self.w_c,
                  self.u_f,self.u_i,self.u_o,self.u_c):
            nn.init.trunc_normal_(
                w,mean=0.0,std=std,a=-3*std,b=3*std
            )
        for b in (self.b_f,self.b_i,self.b_o,self.b_c):
            nn.init.zeros_(b)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            batch_size=x.shape[0]
            h0=torch.zeros(
                batch_size,self.d_model,device=x.device,dtype=x.dtype
            )
            c0=torch.zeros(
                batch_size,self.d_model,device=x.device,dtype=x.dtype
            )
        else:
            h0=state[0]
            c0=state[1]
        f=torch.sigmoid(
            x @ self.w_f.T + h0 @ self.u_f.T + self.b_f
        )
        i=torch.sigmoid(
            x @ self.w_i.T + h0 @ self.u_i.T + self.b_i
        )
        o=torch.sigmoid(
            x @ self.w_o.T + h0 @ self.u_o.T + self.b_o
        )
        g=torch.tanh(
            x @ self.w_c.T + h0 @ self.u_c.T + self.b_c
        )
        c1=f * c0 + i * g
        h1=o * torch.tanh(c1)
        return (h1 , c1)



class LSTM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super(LSTM,self).__init__()
        self.d_model=d_model
        self.num_layers=num_layers
        self.device=device
        self.dtype=dtype
        self.layers=nn.ModuleList([
            LSTMCell(
                d_model,device,dtype
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        batch_size=x.shape[0]
        seq_len=x.shape[1]
        if state is None:
            h=torch.zeros(
                self.num_layers,batch_size,self.d_model,
                device=x.device,dtype=x.dtype
            )
            c=torch.zeros(
                self.num_layers,batch_size,self.d_model,
                device=x.device,dtype=x.dtype
            )
        else:
            h=state[0]
            c=state[1]
        x_set=[]
        for t in range(seq_len):
            x_t=x[:,t,:]
            h_list=[]
            c_list=[]
            for j,layer in enumerate(self.layers):
                h_j,c_j=layer(x_t,(h[j],c[j]))
                x_t=h_j
                h_list.append(h_j)
                c_list.append(c_j)
            h=torch.stack(h_list,dim=0)
            c=torch.stack(c_list,dim=0)
            x_set.append(x_t)
        return torch.stack(x_set,dim=1),(h,c)



class LSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super(LSTMLM,self).__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.num_layers=num_layers
        self.embedding=Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.lstm=LSTM(
            d_model=d_model,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
        self.rmsnorm=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.linear=Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        x=self.embedding(input_ids)
        x,_=self.lstm(x,state)
        h=self.rmsnorm(x)
        logits=self.linear(h)
        return logits

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        end_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        return generate_tokens(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            end_token_id=end_token_id,
        )