# MIT License

# Copyright (c) 2024 Ysobel Sims

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================

# Keeps track of the validation loss and triggers early stopping when applicable
# Patience indicates how many times the loss can get worse before stopping

class EarlyStopping(object):
    def __init__(self, patience=50):
        self.patience = patience  # how many times to let a bad loss through in a row
        self.lowest_loss = None  # previous loss
        self.num_trigger = 0  # how many times a bad loss has happened in a row

    def stop(self, loss):
        # Not seen anything yet, initialise
        if self.lowest_loss == None:
            self.lowest_loss = loss
        # New best loss
        elif loss < self.lowest_loss:
            self.lowest_loss = loss
            self.num_trigger = 0
        # Otherwise add to the trigger
        else:
            self.num_trigger += 1
            # If triggered too many times, stop
            if self.num_trigger >= self.patience:
                return True
        # Survived, keep going!
        return False