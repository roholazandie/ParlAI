from parlai.core.torch_agent import TorchAgent, Output

class ParrotAgent(TorchAgent):
    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        return Output([self.dict.vec2text(row) for row in batch.text_vec])

