import torch
import torch.nn as nn

class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out, padding=1):
        super().__init__()

        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.nn.Conv2d(dim_in,
                                       dim_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)\

    def forward(self, x):
        #x -> [1, 128, 10, 10]

        res = x
        if self.res:
            #[1, 128, 10, 10] -> [1, 256, 10, 10]
            res = self.res(x)

        #[1, 128, 10, 10] -> [1, 256, 10, 10]
        return res + self.s(x)

class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=512,
                                       num_groups=32,
                                       eps=1e-6,
                                       affine=True)

        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        # print("the shape of atten's output is: ", x.shape)
        #x -> [1, 512, 64, 64]
        res = x

        #norm,维度不变
        #[1, 512, 64, 64]
        x = self.norm(x)

        #[1, 512, 64, 64] -> [1, 512, 4096] -> [1, 4096, 512]
        x = x.flatten(start_dim=2).transpose(1, 2)

        #线性运算,维度不变
        #[1, 4096, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        #[1, 4096, 512] -> [1, 512, 4096]
        k = k.transpose(1, 2)

        #[1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        #0.044194173824159216 = 1 / 512**0.5
        #atten = q.bmm(k) * 0.044194173824159216

        #照理来说应该是等价的,但是却有很小的误差
        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
                              q,
                              k,
                              beta=0,
                              alpha=0.044194173824159216)

        atten = torch.softmax(atten, dim=2)

        #[1, 4096, 4096] * [1, 4096, 512] -> [1, 4096, 512]
        atten = atten.bmm(v)

        #线性运算,维度不变
        #[1, 4096, 512]
        atten = self.out(atten)

        #[1, 4096, 512] -> [1, 512, 4096] -> [1, 512, 64, 64]
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)

        #残差连接,维度不变
        #[1, 512, 64, 64]
        atten = atten + res

        return atten

class Pad(torch.nn.Module):
    def __init__(self, dim=(0, 1, 0, 1)):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.pad(x, self.dim,
                                       mode='constant',
                                       value=0)

class singleVAE_73(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            #in
            torch.nn.Conv2d(73, 128, kernel_size=3, stride=1, padding=1),

            #down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, 3, stride=1, padding=1),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            #mid
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            #正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = torch.nn.Sequential(
            #正态分布层
            torch.nn.Conv2d(4, 4, 1),

            #in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            #middle
            torch.nn.Sequential(Resnet(512, 512), Atten(), Resnet(512, 512)),

            #up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=1.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=1.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=1.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=0),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 73, 3, padding=0),
                Pad((0, 1, 0, 1)),
            ),
        )

    def sample(self, h):
        #h -> [1, 8, 64, 64]

        #[1, 4, 64, 64]
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5

        #[1, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h

        return h

    def forward(self, x):
        # print("the shape of input's output is: ", x.shape)
        #x -> [1, 3, 512, 512]

        #[1, 3, 512, 512] -> [1, 8, 64, 64]
        h = self.encoder(x)
        # print("the shape of encoder's output is: ", h.shape)
        #[1, 8, 64, 64] -> [1, 4, 64, 64]
        h = self.sample(h)

        #[1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)
        # print("the shape of decoder's output is: ", h.shape)

        return h

class singleVAE_61(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            #in
            torch.nn.Conv2d(61, 128, kernel_size=3, stride=1, padding=1),

            #down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad((2, 2, 0, 1)),
                    torch.nn.Conv2d(128, 128, 5, stride=1, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad((2, 2, 0, 0)),
                    torch.nn.Conv2d(256, 256, 5, stride=1, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad((2, 3, 0, 0)),
                    torch.nn.Conv2d(512, 512, 3, stride=1, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            #mid
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            #正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = torch.nn.Sequential(
            #正态分布层
            torch.nn.Conv2d(4, 4, 1),

            #in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            #middle
            torch.nn.Sequential(Resnet(512, 512), Atten(), Resnet(512, 512)),

            #up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=1.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                Pad((0, 0, 2, 2)),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=1.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=0),
                Pad((0, 0, 2, 2)),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=1.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=0),
                Pad((0, 0, 2, 2)),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 61, 3, padding=1),
                Pad(),
            ),
        )

    def sample(self, h):
        #h -> [1, 8, 64, 64]

        #[1, 4, 64, 64]

        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5

        #[1, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h

        return h

    def forward(self, x):
        # print("the shape of input's output is: ", x.shape)
        #x -> [1, 3, 512, 512]

        #[1, 3, 512, 512] -> [1, 8, 64, 64]
        h = self.encoder(x)
        # print("the shape of encoder's output is: ", h.shape)
        #[1, 8, 64, 64] -> [1, 4, 64, 64]
        h = self.sample(h)

        #[1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)
        # print("the shape of decoder's output is: ", h.shape)

        return h

class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.VAE_Coronal = singleVAE_73()
        self.VAE_Sagittal = singleVAE_61()
        self.VAE_Axial = singleVAE_61()

    def forward(self, x):

        coronal = x.permute((0, 2, 1, 3)) # 冠状面 [n, 73, 61, 61]
        sagittal = x # 矢状面 [n, 61, 73, 61]
        axial = x.permute((0, 3, 2, 1)) # 横截面 [n, 61, 73, 61]

        # encode
        h_coronal = self.VAE_Coronal.encoder(coronal)
        h_sagittal = self.VAE_Sagittal.encoder(sagittal)
        h_axial = self.VAE_Axial.encoder(axial)

        # print(h_coronal.shape, h_sagittal.shape, h_axial.shape)

        # sample
        h_coronal = self.VAE_Coronal.sample(h_coronal)
        h_sagittal = self.VAE_Sagittal.sample(h_sagittal)
        h_axial = self.VAE_Axial.sample(h_axial)

        # torch.Size([n, 4, 64, 64]) torch.Size([n, 4, 64, 64]) torch.Size([n, 4, 64, 64])
        # print(h_coronal.shape, h_sagittal.shape, h_axial.shape)

        # decode
        coronal = self.VAE_Coronal.decoder(h_coronal)
        sagittal = self.VAE_Sagittal.decoder(h_sagittal)
        axial = self.VAE_Axial.decoder(h_axial)

        # torch.Size([n, 73, 61, 61]) torch.Size([n, 61, 73, 61]) torch.Size([n, 61, 73, 61])
        # print(coronal.shape, sagittal.shape, axial.shape)

        coronal = coronal.permute((0, 2, 1, 3)) # 冠状面 [n, 61, 73, 61]
        axial = axial.permute((0, 3, 2, 1)) # 横截面 [n, 61, 73, 61]

        # combine the three views
        out = (coronal + sagittal + axial) / 3

        return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Test the model
    input_shape = (61, 73, 61)  # Add the channel dimension for PyTorch
    x = torch.randn(4, *input_shape).to(device)

    VAE_model = VAE().to(device)

    # 计算模型参数量
    pytorch_total_params = sum(p.numel() for p in VAE_model.parameters())
    need_gpu_memory = pytorch_total_params * 4 / 1024 / 1024
    print(f"model1参数量: {pytorch_total_params}, 需要显存: {need_gpu_memory}MB")

    out = VAE_model(x)
    print(out.shape)



