import torch
from functorch import vmap, hessian, jacfwd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd.functional import jacobian #, hessian
# based on https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
from torch.distributions import kl


class IterativeLQG:

    def __init__(self, dyn, naf_r, naf_p, sl, al, b_s, t_s, device):
        """
        Args:
            dyn: dynamic
            naf_r: reward
            naf_p: reward based policy
            sl: state length
            al: action length
            b_s: batch size
            t_s: time step
        """

        self.dyn = dyn
        self.sl = sl
        self.al = al
        self.if_conv = 0
        self.NAF_R = naf_r
        self.NAF_P = naf_p
        self.b_s = b_s
        self.ts = t_s
        self.device = device
        self.S = torch.zeros((self.ts, self.b_s, self.sl)).to(self.device)
        self.A = torch.zeros((self.ts, self.b_s, self.al)).to(self.device)
        self.R = torch.empty((self.ts, self.b_s, 1)).to(self.device)
        self.K_arr = torch.zeros(self.ts, self.b_s, self.al, self.sl).to(self.device)
        self.k_arr = torch.zeros(self.ts, self.b_s, 1, self.al).to(self.device)

    def get_global_action(self, state):
        mean, var = self.NAF_P.prob(state)
        m = MultivariateNormal(mean, var)
        return m.sample()

    def get_local_action(self, state):
        state = state.unsqueeze(-2)
        act = torch.rand((1, self.al)).to(self.device)
        mean, var = self.fit(state, act, 1)
        mean = mean.squeeze(-2)
        m = MultivariateNormal(mean, var)
        return m.sample()

    def total_cost(self, sa_in):
        state, action = torch.split(sa_in, [self.sl, self.al], dim=-1)

        mean, var = self.NAF_P.prob(state)
        t_mean, t_var = self.NAF_R.prob(sa_in)
        _input = MultivariateNormal(mean, var)
        _target = MultivariateNormal(t_mean, t_var)
        kld = kl.kl_divergence(_input, _target)

        """
        mean_d = (mean - t_mean).unsqueeze(0)
        mean_d_t = torch.transpose(mean_d, -2, -1)
        kld = torch.log(torch.linalg.det(t_var)) - torch.log(torch.linalg.det(var))
        kld = kld + torch.trace(torch.matmul(torch.linalg.inv(t_var), var))
        kld = kld + torch.matmul(torch.matmul(mean_d, torch.linalg.inv(t_var)), mean_d_t)
        kld = kld.squeeze()
        """

        reward = self.NAF_R.sa_reward(sa_in)

        return reward + (self.lamb * kld)

    def update_lamb(self, lamb):
        self.lamb = lamb

    def _forward(self):

        new_s = torch.zeros((self.ts, self.b_s, self.sl)).to(self.device)
        new_a = torch.zeros((self.ts, self.b_s, self.al)).to(self.device)
        s = self.S[0].clone().detach()

        i = 0
        while i < self.ts:
            new_s[i] = s
            state_difference = (new_s[i] - self.S[i]).unsqueeze(1)

            state_action_trans = torch.matmul(state_difference, torch.transpose((self.K_arr[i]), 1, 2))

            new_a[i] = (state_action_trans + self.k_arr[i]).squeeze(1) + self.A[i]
            sa_in = torch.cat((new_s[i], new_a[i]), dim=1)
            s = self.dyn(sa_in)
            i = i + 1
        self.S = new_s
        self.A = new_a

    def _backward(self):

        _C = torch.zeros(self.b_s, self.al + self.sl, self.al + self.sl).to(self.device)
        _F = torch.zeros(self.b_s, self.sl, self.al + self.sl).to(self.device)
        _c = torch.zeros(self.b_s, 1, self.al + self.sl).to(self.device)
        _V = torch.zeros(self.b_s, self.sl, self.sl).to(self.device)
        _v = torch.zeros(self.b_s, 1, self.sl).to(self.device)
        sa_in = torch.cat((self.S, self.A), dim=-1)

        i = self.ts - 1
        while i > -1:
            j = 0
            print("loop start")
            """
            _C = vmap(hessian(self.total_cost))(sa_in[i])
            _c = vmap(jacfwd(self.total_cost))(sa_in[i])
            _F = vmap(jacfwd(self.dyn))(sa_in[i])
            """
            while j < self.b_s:
                _C[j] = hessian(self.total_cost, sa_in[i][j])
                _c[j] = jacobian(self.total_cost, sa_in[i][j])
                _F[j] = jacobian(self.dyn, sa_in[i][j])
                j = j + 1

            print("loopend so time consumming")
            _transF = torch.transpose(_F, -2, -1)
            _Q = _C + torch.matmul(torch.matmul(_transF, _V), _F)

            # eq 5[c~e]
            _q = _c + torch.matmul(_v, _F)
            # eq 5[a~b]

            _Q_pre1, _Q_pre2 = torch.split(_Q, [self.sl, self.al], dim=-2)
            _Q_xx, _Q_xu = torch.split(_Q_pre1, [self.sl, self.al], dim=-1)
            _Q_ux, _Q_uu = torch.split(_Q_pre2, [self.sl, self.al], dim=-1)

            _Q_x, _Q_u = torch.split(_q, [self.sl, self.al], dim=-1)

            try:
                _invQuu = torch.linalg.inv(_Q_uu - torch.eye(self.al).to(self.device) * 0.01)
                # - torch.eye(self.al)) #regularize term
                # eq [9]
            except:
                _invQuu = torch.linalg.inv(_Q_uu + torch.eye(self.al).to(self.device) * 0.01)
                self.if_conv = 1

            _K = -torch.matmul(_invQuu, _Q_ux)
            _transK = torch.transpose(_K, -2, -1)

            _k = -torch.matmul(_Q_u, _invQuu)
            _V = (_Q_xx + torch.matmul(_Q_xu, _K) +
                  torch.matmul(_transK, _Q_ux) +
                  torch.matmul(torch.matmul(_transK, _Q_uu), _K)
                  )
            # eq 11c
            _v = (_Q_x + torch.matmul(_k, _Q_ux) +
                  torch.matmul(_Q_u, _K) +
                  torch.matmul(_k, torch.matmul(_Q_uu, _K))
                  )
            # eq 11b
            self.K_arr[i] = _K
            self.k_arr[i] = _k
            i = i - 1
        _, cov = torch.split(_C, [self.sl, self.al], dim=-2)
        _, cov = torch.split(cov, [self.sl, self.al], dim=-1)
        return cov.squeeze()

    def fit(self, state, action, batch):
        self.b_s = batch
        self.R = torch.empty((self.ts, self.b_s, 1)).to(self.device)
        self.K_arr = torch.zeros(self.ts, self.b_s, self.al, self.sl).to(self.device)
        self.k_arr = torch.zeros(self.ts, self.b_s, 1, self.al).to(self.device)
        self.S = torch.rand((self.ts, self.b_s, self.sl)).to(self.device)
        self.A = torch.rand((self.ts, self.b_s, self.al)).to(self.device)
        self.A[0] = action
        self.S[0] = state
        _C = None
        i = 0
        while (self.if_conv != 1) and i < 3:
            i = i + 1
            self._forward()
            _C = self._backward()
            print(self.A[0])
        self._forward()

        return self.A[0], _C

