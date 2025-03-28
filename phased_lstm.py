import math

import torch
import torch.nn as nn



class PhasedLSTMCell(nn.Module):
    """Phased LSTM recurrent network cell.
    https://arxiv.org/pdf/1610.09513v1.pdf
    """

    def __init__(
        self,
        hidden_size,
        leak=0.001,
        ratio_on=0.1,
        period_init_min=1.0,
        period_init_max=1000.0
    ):
        """
        Args:
            hidden_size: int, The number of units in the Phased LSTM cell.
            leak: float or scalar float Tensor with value in [0, 1]. Leak applied
                during training.
            ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
                period during which the gates are open.
            period_init_min: float or scalar float Tensor. With value > 0.
                Minimum value of the initialized period.
                The period values are initialized by drawing from the distribution:
                e^U(log(period_init_min), log(period_init_max))
                Where U(.,.) is the uniform distribution.
            period_init_max: float or scalar float Tensor.
                With value > period_init_min. Maximum value of the initialized period.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.ratio_on = ratio_on
        self.leak = leak

        # initialize time-gating parameters
        period = torch.exp(
            torch.Tensor(hidden_size).uniform_(
                math.log(period_init_min), math.log(period_init_max)
            )
        )
        self.tau = nn.Parameter(period)

        phase = torch.Tensor(hidden_size).uniform_() * period
        self.phase = nn.Parameter(phase)

    def _compute_phi(self, t):
        t_ = t.view(-1, 1).repeat(1, self.hidden_size)
        phase_ = self.phase.view(1, -1).repeat(t.shape[0], 1)
        tau_ = self.tau.view(1, -1).repeat(t.shape[0], 1)

        phi = torch.fmod((t_ - phase_), tau_).detach()
        phi = torch.abs(phi) / tau_
        return phi

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return x + (torch.fmod(x, y) - x).detach()

    def set_state(self, c, h):
        self.h0 = h
        self.c0 = c

    def forward(self, c_s, h_s, t):
        # print(c_s.size(), h_s.size(), t.size())
        phi = self._compute_phi(t)

        # Phase-related augmentations
        k_up = 2 * phi / self.ratio_on
        k_down = 2 - k_up
        k_closed = self.leak * phi

        k = torch.where(phi < self.ratio_on, k_down, k_closed)
        k = torch.where(phi < 0.5 * self.ratio_on, k_up, k)
        k = k.view(c_s.shape[0], t.shape[0], -1)

        c_s_new = k * c_s + (1 - k) * self.c0
        h_s_new = k * h_s + (1 - k) * self.h0

        return h_s_new, c_s_new


class PhasedLSTM(nn.Module):
    """Wrapper for multi-layer sequence forwarding via
       PhasedLSTMCell"""

    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional=True
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.bi = 2 if bidirectional else 1

        self.phased_cell = PhasedLSTMCell(
            hidden_size=self.bi * hidden_size
        )

    def forward(self, u_sequence):
        """
        Args:
            sequence: The input sequence data of shape (batch, time, N)
            times: The timestamps corresponding to the data of shape (batch, time)
        """

        c0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size))
        h0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size))
        self.phased_cell.set_state(c0, h0)

        outputs = []
        for i in range(u_sequence.size(1)):
            u_t = u_sequence[:, i, :-1].unsqueeze(1)
            t_t = u_sequence[:, i, -1]

            out, (c_t, h_t) = self.lstm(u_t, (c0, h0))
            (c_s, h_s) = self.phased_cell(c_t, h_t, t_t)

            self.phased_cell.set_state(c_s, h_s)
            c0, h0 = c_s, h_s

            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)

        return outputs