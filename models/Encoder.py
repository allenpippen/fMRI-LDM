import torch
from transformers import CLIPTextModel

class Embed(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = torch.nn.Embedding(49408, 768)
        self.pos_embed = torch.nn.Embedding(77, 768)

        self.register_buffer('pos_ids', torch.arange(77).unsqueeze(dim=0))

    def forward(self, input_ids):
        #input_ids -> [b, 77]

        #[b, 77] -> [b, 77, 768]
        embed = self.embed(input_ids)

        #[1, 77] -> [1, 77, 768]
        pos_embed = self.pos_embed(self.pos_ids)

        #[b, 77, 768]
        return embed + pos_embed

class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(768, 768)
        self.k = torch.nn.Linear(768, 768)
        self.v = torch.nn.Linear(768, 768)
        self.out = torch.nn.Linear(768, 768)

    def forward(self, x):
        #x -> [b, 77, 768]

        b = x.shape[0]

        #维度不变
        #[b, 77, 768]
        q = self.q(x) * 0.125
        k = self.k(x)
        v = self.v(x)

        #拆分注意力头
        #[b, 77, 768] -> [b, 77, 12, 64] -> [b, 12, 77, 64] -> [b*12, 77, 64]
        q = q.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        k = k.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        v = v.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)

        #计算qk乘积
        #[b*12, 77, 64] * [b*12, 64, 77] -> [b*12, 77, 77]
        attn = torch.bmm(q, k.transpose(1, 2))

        #[b*12, 77, 77] -> [b, 12, 77, 77]
        attn = attn.reshape(b, 12, 77, 77)

        #覆盖mask
        def get_mask(b):
            mask = torch.empty(b, 77, 77)

            #上三角的部分置为负无穷
            mask.fill_(-float('inf'))

            #对角线和以下的位置为0
            mask.triu_(1)

            return mask.unsqueeze(1)

        #[b, 12, 77, 77] + [b, 1, 77, 77] -> [b, 12, 77, 77]
        attn = attn + get_mask(attn.shape[0]).to(attn.device)

        #[b, 12, 77, 77] -> [b*12, 77, 77]
        attn = attn.reshape(b * 12, 77, 77)

        #计算softmax,被mask的部分值为0
        attn = attn.softmax(dim=-1)

        #计算和v的乘积
        #[b*12, 77, 77] * [b*12, 77, 64] -> [b*12, 77, 64]
        attn = torch.bmm(attn, v)

        #[b*12, 77, 64] -> [b, 12, 77, 64] -> [b, 77, 12, 64] -> [b, 77, 768]
        attn = attn.reshape(b, 12, 77, 64).transpose(1, 2).reshape(b, 77, 768)

        #线性输出,维度不变
        #[b, 77, 768]
        return self.out(attn)

class ClipEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.s1 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Atten(),
        )

        self.s2 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
        )

        self.s3 = torch.nn.Linear(3072, 768)

    def forward(self, x):
        #x -> [2, 77, 768]

        #维度不变
        #[2, 77, 768]
        x = x + self.s1(x)

        #[2, 77, 768]
        res = x

        #[2, 77, 768] -> [2, 77, 3072]
        x = self.s2(x)

        #维度不变
        #[2, 77, 3072]
        x = x * (x * 1.702).sigmoid()

        #[2, 77, 3072] -> [2, 77, 768]
        return res + self.s3(x)


def load_params_with_hierarchy(file_path):
    # 加载参数
    params = torch.load(file_path)

    # 创建一个新的字典用于存储分离的键
    separated_params = {}

    # 遍历原始参数字典
    for key, value in params.items():
        # 将键按照 '.' 拆分
        keys = key.split('.')

        # 在分离的字典中构建嵌套结构
        d = separated_params
        for k in keys[:-1]:  # 所有部分，除了最后一部分
            if k not in d:
                d[k] = {}
            d = d[k]

        # 最后一个键对应的值
        d[keys[-1]] = value

    return separated_params

def getEncoder():

    encoder = torch.nn.Sequential(
        Embed(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        ClipEncoder(),
        torch.nn.LayerNorm(768),
    )
    #
    # print(encoder(torch.ones(2, 77).long()).shape)

    # 加载预训练模型的参数
    # ./weights/pytorch_model.bin
    params = CLIPTextModel.from_pretrained(
        './weights/diffsion_from_scratch.params', subfolder='text_encoder')
    # params = torch.load('../weights/pytorch_model.bin')
    # file_path = '../weights/diffsion_from_scratch.params/text_encoder/pytorch_model.bin'
    # params = load_params_with_hierarchy(file_path)

    # 词编码
    encoder[0].embed.load_state_dict(
        params.text_model.embeddings.token_embedding.state_dict())

    # 位置编码
    encoder[0].pos_embed.load_state_dict(
        params.text_model.embeddings.position_embedding.state_dict())

    # 12层编码层
    for i in range(12):
        # 第一层norm
        encoder[i + 1].s1[0].load_state_dict(
            params.text_model.encoder.layers[i].layer_norm1.state_dict())

        # 注意力q矩阵
        encoder[i + 1].s1[1].q.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.q_proj.state_dict())

        # 注意力k矩阵
        encoder[i + 1].s1[1].k.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.k_proj.state_dict())

        # 注意力v矩阵
        encoder[i + 1].s1[1].v.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.v_proj.state_dict())

        # 注意力out
        encoder[i + 1].s1[1].out.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.out_proj.state_dict())

        # 第二层norm
        encoder[i + 1].s2[0].load_state_dict(
            params.text_model.encoder.layers[i].layer_norm2.state_dict())

        # mlp第一层fc
        encoder[i + 1].s2[1].load_state_dict(
            params.text_model.encoder.layers[i].mlp.fc1.state_dict())

        # mlp第二层fc
        encoder[i + 1].s3.load_state_dict(
            params.text_model.encoder.layers[i].mlp.fc2.state_dict())

    # 输出norm
    encoder[13].load_state_dict(params.text_model.final_layer_norm.state_dict())

    return encoder

if __name__ == '__main__':

    encoder = getEncoder()

    a = encoder(torch.arange(77).unsqueeze(dim=0))
    print(a.shape)