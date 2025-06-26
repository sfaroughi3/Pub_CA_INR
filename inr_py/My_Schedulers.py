"""Context-Aware Implicit Neural Representations to Compress Earth Systems Model Data

DEVELOPED AT:
                    Department of Chemical Engineering
                    University of Utah, Salt Lake City,
                    Utah 84112, USA
                    DIRECTOR: Prof.  Salah A Faroughi

DEVELOPED BY:
                    Energy and Intelligence Lab

MIT License

Copyright (c) 2024 Energy and Intelligence Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class My_StepLR():
  """
    Custom learning rate scheduler that updates the learning rate every `step_size` steps
    by multiplying it by `gamma`.

    Args:
      lr (float): The initial learning rate.
      step_size (int): Period of learning rate decay.
      gamma (float, optional): Multiplicative factor of learning rate decay. Default is 0.1.

    Attributes:
      step_size (int): Period of learning rate decay.
      gamma (float): Multiplicative factor of learning rate decay.
      lr (float): The current learning rate.
      counter (int): Counter to keep track of steps.
      min_lr (float): Minimum value of the learning rate. Default is 1e-8.

    Methods:
      step(): Increment the step counter.
      update_lr(): Update the learning rate if the counter is a multiple of `step_size`.
      get_lr(): Get the current learning rate, ensuring it is not below `min_lr`.

      Example usage:
        ```
        scheduler = My_StepLR(lr=0.01, step_size=2, gamma=0.1)

        scheduler.step()
        if scheduler.update_lr():
          model.load_state_dict(best_model_state)
          optimizer = optim.Adam(model.parameters(), lr=scheduler.get_lr())
        ```
  """
  def __init__(self, lr, step_size, gamma=0.1) -> None:
    self.step_size = step_size
    self.gamma = gamma
    self.lr = lr
    self.counter = 0
    self.min_lr = 1e-8

  def step(self) -> None:
    """Increment the step counter."""
    self.counter += 1

  def update_lr(self) -> bool:
    """Update the learning rate if the counter is a multiple of `step_size`."""
    if self.counter % self.step_size == 0:
      self.lr *= self.gamma

    return self.counter % self.step_size == 0

  def get_lr(self) -> int:
    """Get the current learning rate, ensuring it is not below `min_lr`."""
    return max(self.lr, self.min_lr)