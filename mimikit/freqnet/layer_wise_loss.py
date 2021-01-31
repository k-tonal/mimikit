import torch

from .freqnet import FreqNet


class LayerWiseLossFreqNet(FreqNet):

    def forward(self, x):
        """
        """
        x = self.inpt(x)
        # we collect all the layers outputs
        skips = None
        outputs = []
        for layer in self.layers:
            x, skips = layer(x, skips)
            # decode each layer's output
            outputs += [self.outpt(skips)]
        if self.training:
            # pass them all to the loss function
            return tuple(outputs)
        # else redirect to a special method for inference
        return self.infer_layer_wise(outputs)
    
    def training_step(self, batch, batch_idx):
        batch = batch[0], batch[1:]
        return super(LayerWiseLossFreqNet, self).training_step(batch, batch_idx)

    def infer_layer_wise(self, outputs):
        """
        Place to implement the inference method when trained layer-wise.
        this default method returns the last output if the network isn't strict, otherwise, it collects the future
        time-steps where they are first predicted in the network (Xt+1 @ layer_1, Xt+2 @ layer_2...)
        @param outputs: list containing all the layers outputs
        @return: the prediction of the network
        """
        if not self.strict:
            return outputs[-1][-1]
        # we collect one future time step pro layer
        rf = self.receptive_field()
        future_steps = []
        for layer_out, shift in zip(outputs, self.all_shifts()):
            # we keep everything from the last layer, hence the condition on the slice
            step = slice(rf - shift, rf - shift +1) if rf - shift > 0 else slice(None, None)
            future_steps += [layer_out[:, step]]
        return torch.cat(future_steps, dim=1)

    def loss_fn(self, predictions, targets):
        denominator = len(self.layers)
        return sum(self._loss_fn(pred, trg) / denominator for pred, trg in zip(predictions, targets))

    def targets_shifts_and_lengths(self, input_length):
        shifts, lengths = self.all_shifts(), self.all_output_lengths(input_length)
        return list(zip(shifts, lengths))